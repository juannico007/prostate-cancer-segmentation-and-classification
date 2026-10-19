from collections import defaultdict
import itk
from torch.nn import utils
from model import UNet
#from .data_augmentation import DataAugmentation, rotate_batch_gpu
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import csv
from dataclasses import dataclass
import os

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

class ProstateDataset(Dataset):
    """Dataset that fetches patient volumes using the records."""

    def __init__(
        self,
        nii_path: Path,
        records: List[Record]
    ):
        self.nii_path = str(nii_path)
        self.records = records
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

        return {
            "patient_id": record.series_id,
            "mri": mri_volume,
            "pet": pet_volume,
            "targets": targets,
            "segmentation": segmentation_target,
            "segmentation_valid": segmentation_valid,
            "patient_score": max(lesion.label for lesion in record.lesions),
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
    segmentation = torch.stack([
        torch.from_numpy(item["segmentation"]).long()
        for item in batch
    ])

    return {
        "patient_id": [item["patient_id"] for item in batch],
        "image": x,
        "targets": [item["targets"] for item in batch],
        "segmentation": segmentation,
        "segmentation_valid": torch.tensor(
            [item["segmentation_valid"] for item in batch], dtype=torch.bool
        ),
        "patient_score": torch.tensor(
            [item["patient_score"] for item in batch], dtype=torch.long
        ),
    }


class TrainingPipeline:
    """Training pipeline for the prostate cancer classification model."""
    
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
        checkpoint_path: str = "prostate_classification_best.pt",
        data_csv: Optional[Path] = None,
        train_ratio: float = 0.8,
        split_seed: int = 42,
        scheduler_config: Optional[Dict[str, Any]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        classification_loss: str = "cross_entropy",
        gaussian_sigma: float = 1.0,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        num_workers: int = 2,
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
            loss_weights: dict with weights for segmentation/classification
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
        self.split_seed = split_seed
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.num_workers = max(0, int(num_workers))
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {
            "name": "step",
            "step_size": 100,
            "gamma": 0.96,
        }
        lw = loss_weights or {}
        self.seg_weight = float(lw.get("segmentation", 1.0))
        self.cls_weight = float(lw.get("classification", 1.0))
        if classification_loss not in {"cross_entropy", "coral", "gaussian"}:
            raise ValueError(
                "classification_loss must be 'cross_entropy', 'coral', or 'gaussian'."
            )
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be greater than zero.")
        self.classification_loss = classification_loss
        self.gaussian_sigma = float(gaussian_sigma)

        self.scheduler_step_mode: Optional[str] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.scheduler_name: Optional[str] = None
        
    def create_dataloaders(
        self,
        nii_path: Path,
        data_csv: Path,
        train_ratio: Optional[float] = None,
        split_seed: Optional[int] = None,
        records: Optional[List[Record]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders from patch metadata.
        """
        if data_csv is None and records is None and self.data_csv is None:
            raise ValueError("data_csv or records must be provided for training.")
        data_csv = data_csv or self.data_csv
        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        split_seed = split_seed if split_seed is not None else self.split_seed

        if records is None:
            records = load_records(Path(data_csv))
        if not records:
            raise ValueError(f"No patch records found in {data_csv}")

        volume_check = ProstateDataset(nii_path=nii_path, records=records)

        rng = np.random.default_rng(split_seed)
        series_ids = sorted({r.series_id for r in records})
        keep_count = max(1, int(round(len(series_ids) * train_ratio)))
        train_series = set(rng.choice(series_ids, size=keep_count, replace=False))
        train_records = [r for r in records if r.series_id in train_series]
        val_records = [r for r in records if r.series_id not in train_series]
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
        #For now we won't add data augmentation, but it will come as a TODO

        print("training records:", [i.series_id for i in train_records])
        print("validation records:", [i.series_id for i in val_records])

        train_dataset = ProstateDataset(
            nii_path=nii_path,
            records=train_records,
        )
        val_dataset = ProstateDataset(
            nii_path=nii_path,
            records=val_records,
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
        segmentation_logits = outputs["segmentation_logits"].float()
        severity_logits = outputs["severity_logits"].float()
        segmentation_targets = batch["segmentation"].to(self.device, non_blocking=True)
        segmentation_valid = batch["segmentation_valid"].to(self.device, non_blocking=True)
        patient_scores = batch["patient_score"].to(self.device, non_blocking=True)

        if segmentation_valid.any():
            valid_logits = segmentation_logits[segmentation_valid]
            valid_targets = segmentation_targets[segmentation_valid]
            cross_entropy = F.cross_entropy(valid_logits, valid_targets)
            probabilities = torch.softmax(valid_logits, dim=1)
            dice_scores = []
            for class_index in range(1, self.model.out_channels):
                class_targets = valid_targets == class_index
                if class_targets.any():
                    class_probabilities = probabilities[:, class_index]
                    intersection = (class_probabilities * class_targets).sum()
                    denominator = class_probabilities.sum() + class_targets.sum()
                    dice_scores.append((2 * intersection + 1e-5) / (denominator + 1e-5))
            dice = torch.stack(dice_scores).mean() if dice_scores else cross_entropy.new_tensor(1.0)
            segmentation_loss = cross_entropy + (1 - dice)
        else:
            segmentation_loss = segmentation_logits.sum() * 0
            dice = segmentation_loss.detach()

        severity_targets = patient_scores - 1
        if self.classification_loss == "coral":
            threshold_count = self.model.num_severity_levels - 1
            thresholds = torch.arange(threshold_count, device=self.device)
            ordinal_targets = (severity_targets.unsqueeze(1) > thresholds).float()
            if severity_logits.shape[1] != threshold_count:
                raise ValueError(
                    f"CORAL expects {threshold_count} threshold logits, "
                    f"received {severity_logits.shape[1]}."
                )
            severity_loss = F.binary_cross_entropy_with_logits(
                severity_logits,
                ordinal_targets,
            )
            severity_predictions = (severity_logits > 0).sum(dim=1) + 1
        elif self.classification_loss == "gaussian":
            class_indices = torch.arange(
                self.model.num_severity_levels,
                device=self.device,
                dtype=severity_logits.dtype,
            )
            gaussian_targets = torch.exp(
                -0.5
                * ((class_indices.unsqueeze(0) - severity_targets.unsqueeze(1))
                   / self.gaussian_sigma) ** 2
            )
            gaussian_targets = gaussian_targets / gaussian_targets.sum(
                dim=1,
                keepdim=True,
            )
            severity_loss = -(
                gaussian_targets * F.log_softmax(severity_logits, dim=1)
            ).sum(dim=1).mean()
            severity_predictions = severity_logits.argmax(dim=1) + 1
        else:
            severity_loss = F.cross_entropy(severity_logits, severity_targets)
            severity_predictions = severity_logits.argmax(dim=1) + 1

        severity_mae = (severity_predictions - patient_scores).abs().float().mean()
        severity_accuracy = (severity_predictions == patient_scores).float().mean()
        total_loss = self.seg_weight * segmentation_loss + self.cls_weight * severity_loss
        return {
            "total": total_loss,
            "segmentation": segmentation_loss,
            "severity": severity_loss,
            "dice": dice,
            "severity_mae": severity_mae,
            "severity_accuracy": severity_accuracy,
        }

    def train(self) -> dict:
        """Train full-volume segmentation and patient-level ordinal severity heads."""
        if not hasattr(self, "train_loader") or not hasattr(self, "val_loader"):
            raise ValueError("Create dataloaders before calling train().")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        use_mixed_precision = self.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self._configure_scheduler()

        metric_names = ("total", "segmentation", "severity", "dice", "severity_mae", "severity_accuracy")
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
                images = batch["image"].to(self.device, non_blocking=True)
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
                f"(seg {train_metrics['segmentation']:.4f}, "
                f"severity {train_metrics['severity']:.4f}, Dice {train_metrics['dice']:.4f}, "
                f"score acc {train_metrics['severity_accuracy']:.3f}) | "
                f"val loss {val_metrics['total']:.4f} "
                f"(seg {val_metrics['segmentation']:.4f}, "
                f"severity {val_metrics['severity']:.4f}, Dice {val_metrics['dice']:.4f}, "
                f"score MAE {val_metrics['severity_mae']:.3f}, "
                f"score acc {val_metrics['severity_accuracy']:.3f})"
            )

            val_loss = val_metrics["total"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)
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