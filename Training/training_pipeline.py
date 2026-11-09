from collections import defaultdict
import itk
from torch.nn import utils
from model import UNet
#from .data_augmentation import DataAugmentation, rotate_batch_gpu
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import csv
from dataclasses import dataclass
import os
from loss_functions import TrainingLosses


@dataclass
class LesionRecord:
    lesion_id: int
    label: int
    x_min: int|None
    y_min: int|None
    z_min: int|None
    x_max: int|None
    y_max: int|None
    z_max: int|None


@dataclass
class Record:
    series_id: int
    lesions: List[LesionRecord]

def load_records(data_csv: Path) -> List[Record]:
    """Load metadata from CSV into Record objects."""
    grouped = defaultdict(list)

    with data_csv.open() as f:
        reader = csv.DictReader(f)
        required = {"study_id", 
                    "PRIMARY", 
                    "lesion_id"}
        
        missing = required.difference(reader.fieldnames or set())
        if missing:
            raise ValueError(f"Data CSV missing columns: {', '.join(sorted(missing))}")
        
        for row in reader:
            try:
                study_id = int(row["study_id"])
                lesion_id = int(row["lesion_id"])
                label = int(row["PRIMARY"])
                if row["x_min"] != "":
                    x_min = int(row["x_min"])
                    y_min = int(row["y_min"])
                    z_min = int(row["z_min"])
                    x_max = int(row["x_max"])
                    y_max = int(row["y_max"])
                    z_max = int(row["z_max"])
                else:
                    x_min = None
                    y_min = None
                    z_min = None
                    x_max = None
                    y_max = None
                    z_max = None

            except (TypeError, ValueError):
                # Skip rows that look like headers or malformed entries
                print(f"[Warning] Skipping malformed row in patient CSV: {row}")
                continue

            grouped[study_id].append(
                    LesionRecord(
                        lesion_id=lesion_id,
                        label=label,
                        x_min=x_min,
                        y_min=y_min,
                        z_min=z_min,
                        x_max=x_max,
                        y_max=y_max,
                        z_max=z_max
                    )
                )
        records = [
            Record(
                series_id=study_id,
                lesions=lesions
            )
            for study_id, lesions in grouped.items()
        ]
    return records


def split_records(
    records: List[Record],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> Dict[str, List[Record]]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("train_ratio, validation_ratio, and test_ratio must be positive and sum to 1.")

    series_ids = sorted({record.series_id for record in records})
    if len(series_ids) < 3:
        raise ValueError("At least three studies are required for train/validation/test splits.")

    rng = np.random.default_rng(split_seed)
    train_count = max(1, int(round(len(series_ids) * train_ratio)))
    train_ids = set(rng.choice(series_ids, size=train_count, replace=False))
    remaining_ids = sorted(set(series_ids) - train_ids)
    validation_count = max(1, int(round(len(series_ids) * validation_ratio)))
    if validation_count >= len(remaining_ids):
        raise ValueError("Not enough studies remain to create non-empty validation and test splits.")
    validation_ids = set(rng.choice(remaining_ids, size=validation_count, replace=False))
    test_ids = set(remaining_ids) - validation_ids

    return {
        "train": [record for record in records if record.series_id in train_ids],
        "validation": [record for record in records if record.series_id in validation_ids],
        "test": [record for record in records if record.series_id in test_ids],
    }


def kfold_split_records(
    records: List[Record],
    k_folds: int,
    split_seed: int,
    test_records: Optional[List[Record]] = None,
) -> List[Dict[str, List[Record]]]:
    """Split the non-test studies into k folds, stratified by each study's highest lesion label.

    Each returned fold uses one partition for validation and the remaining k-1 for training;
    the held-out test records are shared by every fold.
    """
    if not isinstance(k_folds, int) or k_folds < 2:
        raise ValueError("k_folds must be an integer greater than or equal to 2.")
    test_records = list(test_records or [])
    test_ids = {record.series_id for record in test_records}
    pool = [record for record in records if record.series_id not in test_ids]
    pool_ids = sorted({record.series_id for record in pool})
    if len(pool_ids) < k_folds:
        raise ValueError(
            f"Cannot create {k_folds} folds from {len(pool_ids)} non-test studies."
        )

    strata = defaultdict(list)
    for record in pool:
        strata[max((lesion.label for lesion in record.lesions), default=0)].append(record.series_id)

    rng = np.random.default_rng(split_seed)
    fold_ids: List[set] = [set() for _ in range(k_folds)]
    offset = 0
    for stratum in sorted(strata):
        ids = sorted(set(strata[stratum]))
        rng.shuffle(ids)
        for position, series_id in enumerate(ids):
            fold_ids[(offset + position) % k_folds].add(series_id)
        offset = (offset + len(ids)) % k_folds

    folds = []
    for validation_ids in fold_ids:
        folds.append({
            "train": [record for record in pool if record.series_id not in validation_ids],
            "validation": [record for record in pool if record.series_id in validation_ids],
            "test": test_records,
        })
    return folds


def summarize_fold_histories(fold_histories: Dict[int, Dict[str, List[float]]]) -> Dict[str, Any]:
    """Summarize per-fold training curves at the checkpointed (lowest validation loss) epoch."""
    per_fold = {}
    for fold, history in sorted(fold_histories.items()):
        val_total = history.get("val_total") or []
        if not val_total:
            continue
        # Training checkpoints on strict improvement, so the first minimum is the saved epoch.
        best_index = int(np.argmin(val_total))
        per_fold[str(fold)] = {
            "epochs_trained": len(val_total),
            "best_epoch": best_index + 1,
            "train_dice_at_best_epoch": float(history["train_dice"][best_index]),
            "val_dice_at_best_epoch": float(history["val_dice"][best_index]),
            "val_loss_at_best_epoch": float(val_total[best_index]),
            "max_val_dice": float(np.max(history["val_dice"])),
            "final_train_dice": float(history["train_dice"][-1]),
            "final_val_dice": float(history["val_dice"][-1]),
        }

    aggregate = {}
    if per_fold:
        for key in next(iter(per_fold.values())):
            if key in {"best_epoch", "epochs_trained"}:
                continue
            values = np.array([fold_summary[key] for fold_summary in per_fold.values()])
            aggregate[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return {"completed_folds": len(per_fold), "folds": per_fold, "aggregate": aggregate}


def write_split_csvs(data_csv: Path, splits: Dict[str, List[Record]], output_dir: Path):
    split_ids = {
        split_name: {record.series_id for record in split_records}
        for split_name, split_records in splits.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_dir / "train_split.csv",
        "validation": output_dir / "validation_split.csv",
        "test": output_dir / "test_split.csv",
    }
    with data_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "study_id" not in reader.fieldnames:
            raise ValueError(f"Data CSV must contain a study_id column: {data_csv}")
        streams = {name: path.open("w", newline="", encoding="utf-8") for name, path in output_paths.items()}
        try:
            writers = {
                name: csv.DictWriter(stream, fieldnames=reader.fieldnames)
                for name, stream in streams.items()
            }
            for writer in writers.values():
                writer.writeheader()
            for row in reader:
                try:
                    series_id = int(row["study_id"])
                except (TypeError, ValueError):
                    continue
                for split_name, ids in split_ids.items():
                    if series_id in ids:
                        writers[split_name].writerow(row)
                        break
        finally:
            for stream in streams.values():
                stream.close()
    return output_paths

class ProstateDataset(Dataset):
    """Dataset that fetches patient volumes using the records."""

    def __init__(
        self,
        nii_path: Path,
        records: List[Record],
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        binary_segmentation: bool = False,
    ):
        self.nii_path = str(nii_path)
        self.records = records
        self.binary_segmentation = binary_segmentation
        if normalization_stats is not None:
            validated_stats = {}
            for modality in ("mri", "pet"):
                modality_stats = normalization_stats.get(modality)
                if not isinstance(modality_stats, dict):
                    raise ValueError(
                        f"Normalization parameters are missing for {modality.upper()}."
                    )
                try:
                    mean = float(modality_stats["mean"])
                    std = float(modality_stats["std"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid normalization parameters for {modality.upper()}."
                    ) from error
                if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                    raise ValueError(
                        f"Normalization mean/std for {modality.upper()} must be finite "
                        "and std must be greater than zero."
                    )
                validated_stats[modality] = {"mean": mean, "std": std}
            self.normalization_stats = validated_stats
        else:
            self.normalization_stats = None
    def __len__(self):
        return len(self.records)

    @staticmethod
    def _find_volume_paths(patient_path):
        pet_image_paths = list(patient_path.rglob("*PET.nii.gz"))
        mri_image_paths = list(patient_path.rglob("*MRI_T2.nii.gz"))
        return pet_image_paths, mri_image_paths

    def has_required_volumes(self, record):
        patient_path = Path(self.nii_path) / str(record.series_id)
        pet_paths, mri_paths = self._find_volume_paths(patient_path)
        return len(pet_paths) == 1 and len(mri_paths) == 1

    def _get_volumes(self, patient_path):
        pet_image_paths, mri_image_paths = self._find_volume_paths(patient_path)
        if len(pet_image_paths) != 1 or len(mri_image_paths) != 1:
            raise FileNotFoundError(
                f"Patient {patient_path.name} requires exactly one PET and one MRI_T2 volume; "
                f"found {len(pet_image_paths)} PET and {len(mri_image_paths)} MRI_T2 files"
            )
        pet_image_path = pet_image_paths[0]
        mri_image_path = mri_image_paths[0]

        pet_image = itk.array_from_image(itk.imread(pet_image_path))
        mri_image = itk.array_from_image(itk.imread(mri_image_path))

        return pet_image, mri_image

    def compute_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate voxel-weighted population mean and standard deviation per modality."""
        accumulators = {
            "mri": {"count": 0, "mean": 0.0, "m2": 0.0},
            "pet": {"count": 0, "mean": 0.0, "m2": 0.0},
        }
        for record in self.records:
            patient_path = Path(self.nii_path) / str(record.series_id)
            pet_volume, mri_volume = self._get_volumes(patient_path)
            for modality, volume in (("mri", mri_volume), ("pet", pet_volume)):
                values = np.asarray(volume, dtype=np.float64)
                if values.size == 0 or not np.isfinite(values).all():
                    raise ValueError(
                        f"Patient {record.series_id} has empty or non-finite {modality.upper()} data."
                    )

                batch_count = int(values.size)
                batch_mean = float(values.mean())
                batch_m2 = float(values.var() * batch_count)
                accumulator = accumulators[modality]
                total_count = accumulator["count"] + batch_count
                delta = batch_mean - accumulator["mean"]
                accumulator["m2"] += (
                    batch_m2
                    + delta * delta * accumulator["count"] * batch_count / total_count
                )
                accumulator["mean"] += delta * batch_count / total_count
                accumulator["count"] = total_count

        stats = {}
        for modality, accumulator in accumulators.items():
            if accumulator["count"] == 0:
                raise ValueError(f"No training {modality.upper()} voxels were available for normalization.")
            std = float(np.sqrt(accumulator["m2"] / accumulator["count"]))
            stats[modality] = {
                "mean": float(accumulator["mean"]),
                "std": std if std > 0 else 1.0,
            }
        return stats

    def _get_lesion_mask(self, patient_path, lesion_id):
        lesion_paths = list(
            patient_path.rglob(f"*Lesion_{lesion_id}.nii.gz")
        )

        if len(lesion_paths) != 1:
            raise ValueError(
                f"Expected exactly one mask for lesion {lesion_id}, "
                f"found {len(lesion_paths)}"
            )

        mask = itk.array_from_image(
            itk.imread(lesion_paths[0])
        )

        return mask

    def __getitem__(self, idx):
        record = self.records[idx]
        patient = str(record.series_id)

        patient_path = Path(self.nii_path) / patient

        pet_volume, mri_volume = self._get_volumes(patient_path)
        if self.normalization_stats is not None:
            mri_stats = self.normalization_stats["mri"]
            pet_stats = self.normalization_stats["pet"]
            mri_volume = (
                np.asarray(mri_volume, dtype=np.float32) - mri_stats["mean"]
            ) / mri_stats["std"]
            pet_volume = (
                np.asarray(pet_volume, dtype=np.float32) - pet_stats["mean"]
            ) / pet_stats["std"]

        targets = []
        segmentation_target = np.zeros(mri_volume.shape, dtype=np.int64)
        segmentation_valid = True

        for lesion in record.lesions:

            #Checks if the segmentation exists, it does it by looking at the
            #coordinates of the bounding box, as the coordinates were taken from it
            has_bbox = all(
                value is not None
                for value in (
                    lesion.x_min,
                    lesion.y_min,
                    lesion.z_min,
                    lesion.x_max,
                    lesion.y_max,
                    lesion.z_max,
                )
            )
            if has_bbox and lesion.label >= 3:
                try:
                    mask = self._get_lesion_mask(
                        patient_path,
                        lesion.lesion_id
                    )
                except ValueError:
                    mask = None
            else:
                mask = None

            if lesion.label >= 3:
                if mask is None:
                    segmentation_valid = False
                else:
                    mask_array = np.asarray(mask) > 0
                    if mask_array.shape != mri_volume.shape:
                        raise ValueError(
                            f"Patient {patient}: lesion {lesion.lesion_id} mask shape "
                            f"{mask_array.shape} does not match image shape {mri_volume.shape}"
                        )
                    np.maximum(
                        segmentation_target,
                        mask_array.astype(np.int64) * (lesion.label - 2),
                        out=segmentation_target,
                    )

            targets.append({
                "lesion_id": lesion.lesion_id,
                "label": lesion.label,
                "mask": mask,
                "bbox": np.array([
                    lesion.z_min,
                    lesion.y_min,
                    lesion.x_min,
                    lesion.z_max,
                    lesion.y_max,
                    lesion.x_max,
                ]) if has_bbox else None,
            })

        if self.binary_segmentation:
            segmentation_target = (segmentation_target > 0).astype(np.int64)

        return {
            "patient_id": record.series_id,
            "mri": mri_volume,
            "pet": pet_volume,
            "targets": targets,
            "segmentation": segmentation_target,
            "segmentation_valid": segmentation_valid,
            "binary_segmentation": self.binary_segmentation,
        }

def collate_fn(batch):
    mri = torch.stack([
        torch.from_numpy(item["mri"]).float()
        for item in batch
    ])

    pet = torch.stack([
        torch.from_numpy(item["pet"]).float()
        for item in batch
    ])

    x = torch.stack([mri, pet], dim=1)
    binary_segmentation = batch[0]["binary_segmentation"]
    if any(item["binary_segmentation"] != binary_segmentation for item in batch):
        raise ValueError("A batch cannot mix binary and multiclass segmentation targets.")
    segmentation = torch.stack([
        torch.from_numpy(item["segmentation"]).float()
        if binary_segmentation
        else torch.from_numpy(item["segmentation"]).long()
        for item in batch
    ])
    if binary_segmentation:
        segmentation = segmentation.unsqueeze(1)

    return {
        "patient_id": [item["patient_id"] for item in batch],
        "image": x,
        "targets": [item["targets"] for item in batch],
        "segmentation": segmentation,
        "segmentation_valid": torch.tensor(
            [item["segmentation_valid"] for item in batch], dtype=torch.bool
        ),
    }


class TrainingPipeline:
    """Training pipeline for the prostate cancer segmentation model."""
    
    def __init__(
        self,
        model: UNet,
        batch_size: int = 4,
        epochs: int = 4,
        learning_rate : float = 1e-4,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 40,
        lr_reduction_patience: int = 10,
        lr_reduction_factor : float = 0.5,
        min_lr : float = 1e-7,
        checkpoint_path: str = "prostate_segmentation_best.pt",
        data_csv: Optional[Path] = None,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 42,
        scheduler_config: Optional[Dict[str, Any]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        num_workers: int = 2,
        class_weight_power: Optional[float] = 2.0,
        grouped_normalization: bool = True,
        binary_segmentation: bool = False,
        data_augmentation: bool = True,
        ):
        """
        Initialize training pipeline.
        
        Parameters
        ----------
            model: ProstateCancerClassificationModel instance
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Initial learning rate for the model
            early_stopping_patience: Number of epochs of no improving until early stopping
            lr_reduction_patience: Number of epochs of plateau until reducing lr
            lr_reduction_factor: Learning rate reduction on plateau
            min_lr: Minimum value for the learning rate
            checkpoint_path: Path to save best model
            loss_weights: dict with the segmentation loss weight
            data_augmentation: Apply random right-angle rotations and axis reflections during training
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.lr_reduction_patience = lr_reduction_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.data_csv = Path(data_csv) if data_csv else None
        self.checkpoint_path = checkpoint_path
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.num_workers = max(0, int(num_workers))
        if not isinstance(grouped_normalization, bool):
            raise ValueError("grouped_normalization must be a boolean.")
        self.grouped_normalization = grouped_normalization
        if not isinstance(binary_segmentation, bool):
            raise ValueError("binary_segmentation must be a boolean.")
        if binary_segmentation and model.out_channels != 1:
            raise ValueError("Binary segmentation requires a model with out_channels=1.")
        self.binary_segmentation = binary_segmentation
        if not isinstance(data_augmentation, bool):
            raise ValueError("data_augmentation must be a boolean.")
        self.data_augmentation = data_augmentation
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {
            "name": "step",
            "step_size": 100,
            "gamma": 0.96,
        }
        self.losses = TrainingLosses(
            segmentation_classes=model.out_channels,
            loss_weights=loss_weights,
            class_weight_power=class_weight_power,
            binary_segmentation=binary_segmentation,
        )
        self.normalization_stats: Optional[Dict[str, Dict[str, float]]] = None

        self.scheduler_step_mode: Optional[str] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.scheduler_name: Optional[str] = None
        
    def create_dataloaders(
        self,
        nii_path: Path,
        data_csv: Path,
        train_ratio: Optional[float] = None,
        validation_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        split_seed: Optional[int] = None,
        records: Optional[List[Record]] = None,
        splits: Optional[Dict[str, List[Record]]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders from patch metadata.
        """
        if data_csv is None and records is None and self.data_csv is None:
            raise ValueError("data_csv or records must be provided for training.")
        data_csv = data_csv or self.data_csv
        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        validation_ratio = validation_ratio if validation_ratio is not None else self.validation_ratio
        test_ratio = test_ratio if test_ratio is not None else self.test_ratio
        split_seed = split_seed if split_seed is not None else self.split_seed

        if records is None:
            records = load_records(Path(data_csv))
        if not records:
            raise ValueError(f"No patch records found in {data_csv}")

        volume_check = ProstateDataset(
            nii_path=nii_path,
            records=records,
            binary_segmentation=self.binary_segmentation,
        )

        if splits is None:
            splits = split_records(
                records,
                train_ratio,
                validation_ratio,
                test_ratio,
                split_seed,
            )
        train_records = splits["train"]
        val_records = splits["validation"]
        missing_train = [r.series_id for r in train_records if not volume_check.has_required_volumes(r)]
        missing_val = [r.series_id for r in val_records if not volume_check.has_required_volumes(r)]
        train_records = [r for r in train_records if r.series_id not in set(missing_train)]
        val_records = [r for r in val_records if r.series_id not in set(missing_val)]
        if missing_train or missing_val:
            print(
                "[Warning] Excluding studies without exactly one PET and one MRI_T2 volume: "
                f"train={missing_train}, validation={missing_val}"
            )
        if not train_records:
            raise ValueError("No training studies have both PET and MRI_T2 volumes.")
        if not val_records:
            raise ValueError("No validation studies have both PET and MRI_T2 volumes.")
        print("training records:", [i.series_id for i in train_records])
        print("validation records:", [i.series_id for i in val_records])

        train_dataset = ProstateDataset(
            nii_path=nii_path,
            records=train_records,
            binary_segmentation=self.binary_segmentation,
        )
        if self.grouped_normalization:
            self.normalization_stats = train_dataset.compute_normalization_stats()
            train_dataset.normalization_stats = self.normalization_stats
        else:
            self.normalization_stats = None
        if (
            not self.binary_segmentation
            and self.losses.class_weight_power is not None
        ):
            self._configure_class_weights(train_dataset)
        val_dataset = ProstateDataset(
            nii_path=nii_path,
            records=val_records,
            normalization_stats=self.normalization_stats,
            binary_segmentation=self.binary_segmentation,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_fn
        )
        self.train_loader, self.val_loader = train_loader, val_loader
        return self.train_loader, self.val_loader

    def _configure_class_weights(self, train_dataset):
        segmentation_counts = np.zeros(self.model.out_channels, dtype=np.int64)
        valid_segmentation_patients = 0
        for index in tqdm(range(len(train_dataset)), desc="class-weight scan", leave=False):
            sample = train_dataset[index]
            if not sample["segmentation_valid"]:
                continue
            counts = np.bincount(
                sample["segmentation"].reshape(-1),
                minlength=self.model.out_channels,
            )
            if len(counts) > self.model.out_channels:
                raise ValueError("Segmentation target contains a class outside model.out_channels.")
            segmentation_counts += counts
            valid_segmentation_patients += 1

        self.losses.configure_class_weights(segmentation_counts)
        print(
            "[Info] Inverse-frequency segmentation weights:",
            self.losses.segmentation_class_weights.tolist(),
            "from voxel counts:",
            segmentation_counts.tolist(),
            f"({valid_segmentation_patients} valid training masks)",
        )

    def _configure_scheduler(self):
        """Create the requested LR scheduler and remember how it should be stepped."""
        config = dict(self.scheduler_config) if self.scheduler_config else {}
        name = config.get("name", "step")
        if not name:
            self.scheduler = None
            self.scheduler_step_mode = None
            self.scheduler_name = None
            return

        name = name.lower()
        self.scheduler_name = name
        if name in {"none", "off"}:
            self.scheduler = None
            self.scheduler_step_mode = None
            return

        if name == "step":
            step_size = int(config.get("step_size", 100))
            gamma = float(config.get("gamma", 0.96))
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
            self.scheduler_step_mode = "epoch"
        elif name == "cosine":
            t_max = int(config.get("T_max", self.epochs))
            eta_min = float(config.get("eta_min", 1e-6))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
            self.scheduler_step_mode = "epoch"
        elif name in {"plateau", "reduce_on_plateau"}:
            factor = float(config.get("factor", 0.5))
            patience = int(config.get("patience", 10))
            min_lr = float(config.get("min_lr", 1e-6))
            mode = config.get("mode", "min")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
            self.scheduler_step_mode = "plateau"
        elif name in {"onecycle", "one_cycle"}:
            if not hasattr(self, "train_loader"):
                raise ValueError("OneCycleLR scheduler requires dataloaders to be created before training.")
            steps_per_epoch = int(config.get("steps_per_epoch", len(self.train_loader)))
            epochs = int(config.get("epochs", self.epochs))
            max_lr = float(config.get("max_lr", self.learning_rate))
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            )
            self.scheduler_step_mode = "batch"
        else:
            raise ValueError(f"Unsupported scheduler '{name}'.")
    
    def _compute_losses(self, outputs, batch):
        return self.losses(outputs, batch)

    @staticmethod
    def _augment_training_data(images, segmentation):
        """Apply matching, shape-preserving spatial transforms independently per sample."""
        augmented_images = []
        augmented_targets = []
        for image, target in zip(images, segmentation):
            for axes in ((-3, -2), (-3, -1), (-2, -1)):
                quarter_turns = int(torch.randint(4, ()).item())
                if image.shape[axes[0]] != image.shape[axes[1]]:
                    quarter_turns = (quarter_turns % 2) * 2
                image = torch.rot90(image, quarter_turns, dims=axes)
                target = torch.rot90(target, quarter_turns, dims=axes)
            flip_axes = tuple(
                axis for axis in (-3, -2, -1) if torch.rand(()).item() < 0.5
            )
            if flip_axes:
                image = torch.flip(image, dims=flip_axes)
                target = torch.flip(target, dims=flip_axes)
            augmented_images.append(image)
            augmented_targets.append(target)
        return torch.stack(augmented_images), torch.stack(augmented_targets)

    def train(self) -> dict:
        """Train the full-volume segmentation model."""
        if not hasattr(self, "train_loader") or not hasattr(self, "val_loader"):
            raise ValueError("Create dataloaders before calling train().")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.losses.to(self.device)
        use_mixed_precision = self.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self._configure_scheduler()

        metric_names = ("total", "segmentation", "dice")
        history = {f"{split}_{name}": [] for split in ("train", "val") for name in metric_names}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        def run_epoch(loader, training):
            self.model.train(training)
            totals = {name: [] for name in metric_names}
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            loader_length = len(loader)
            split_name = "train" if training else "validation"
            for batch_index, batch in enumerate(tqdm(loader, leave=False, desc=split_name)):
                if not batch["segmentation_valid"].any():
                    continue
                images = batch["image"].to(self.device, non_blocking=True)
                if training and self.data_augmentation:
                    segmentation = batch["segmentation"].to(self.device, non_blocking=True)
                    images, segmentation = self._augment_training_data(images, segmentation)
                    batch = {**batch, "segmentation": segmentation}
                with torch.set_grad_enabled(training):
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=use_mixed_precision,
                    ):
                        outputs = self.model(images)
                        losses = self._compute_losses(outputs, batch)

                    if training:
                        group_start = (
                            batch_index // self.gradient_accumulation_steps
                        ) * self.gradient_accumulation_steps
                        group_size = min(
                            self.gradient_accumulation_steps,
                            loader_length - group_start,
                        )
                        scaler.scale(losses["total"] / group_size).backward()
                        is_update_step = (
                            (batch_index + 1) % self.gradient_accumulation_steps == 0
                            or batch_index + 1 == loader_length
                        )
                        if is_update_step:
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                for name in metric_names:
                    totals[name].append(float(losses[name].detach().cpu()))
            if not totals["total"]:
                raise ValueError(f"No valid segmentation targets found in {split_name} data.")
            return {
                name: float(np.mean(values)) if values else 0.0
                for name, values in totals.items()
            }

        for epoch in range(1, self.epochs + 1):
            train_metrics = run_epoch(self.train_loader, training=True)
            with torch.no_grad():
                val_metrics = run_epoch(self.val_loader, training=False)

            for name in metric_names:
                history[f"train_{name}"].append(train_metrics[name])
                history[f"val_{name}"].append(val_metrics[name])

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"train loss {train_metrics['total']:.4f} "
                f"(seg {train_metrics['segmentation']:.4f}, Dice {train_metrics['dice']:.4f}) | "
                f"val loss {val_metrics['total']:.4f} "
                f"(seg {val_metrics['segmentation']:.4f}, Dice {val_metrics['dice']:.4f})"
            )

            val_loss = val_metrics["total"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "normalization_stats": self.normalization_stats,
                    },
                    self.checkpoint_path,
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.scheduler:
                if self.scheduler_step_mode == "epoch":
                    self.scheduler.step()
                elif self.scheduler_step_mode == "plateau":
                    self.scheduler.step(val_loss)

            if epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return history