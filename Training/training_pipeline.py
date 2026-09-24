import itk
from torch.nn import utils
from model import UNet
#from .data_augmentation import DataAugmentation, rotate_batch_gpu
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import csv
from dataclasses import dataclass
import os

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply sigmoid if preds are raw logits
        preds = torch.sigmoid(preds)
        
        # Flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1. - dice_coeff

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: List[str]):
        super().__init__()
        self.tasks = tasks
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in tasks
        })

    def forward(self, loss_dict: Dict[str, torch.Tensor]):
        total_loss = 0
        for task, loss in loss_dict.items():
            if task not in self.log_vars:
                continue
            log_var = self.log_vars[task]
            precision = torch.exp(-log_var)
            total_loss += 0.5 * precision * loss + 0.5 * log_var
        return total_loss

@dataclass
class Record:
    series_id: int
    lesion_id: int
    label: int

def load_patch_records(patch_csv: Path) -> List[Record]:
    """Load patch metadata from CSV into Record objects."""
    records: List[Record] = []
    with patch_csv.open() as f:
        reader = csv.DictReader(f)
        required = {"study_id", "PRIMARY", "lesion_id"}
        missing = required.difference(reader.fieldnames or set())
        if missing:
            raise ValueError(f"Data CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                # TODO: Have to modify this to account for multiple labels per patient
                labels = int(row["PRIMARY"])
                lesion_ids = int(row["lesion_id"])
            except (TypeError, ValueError):
                # Skip rows that look like headers or malformed entries
                print(f"[Warning] Skipping malformed row in patient CSV: {row}")
                continue
            records.append(
                Record(
                    series_id=int(row["study_id"]),
                    label=labels,
                    lesion_id=lesion_ids
                )
            )
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

    def _get_volumes(self, patient):        
        possible_pet_image_path = Path(os.path.join(self.nii_path, patient)).rglob("*PET-PSMA-PROSTATA_study*.nii.gz")
        pet_image_path = next(p for p in possible_pet_image_path
                                if "Lesion" not in p.name)
        mri_image_path = next(Path(os.path.join(self.nii_path, patient)).rglob("*MRI-PROSTATA_study*.nii.gz"), None)
        pet_image = itk.array_from_image(itk.imread(pet_image_path))
        mri_image = itk.array_from_image(itk.imread(mri_image_path))

        lesion_image_paths = list(Path(os.path.join(self.nii_path, patient)).rglob("*Lesion*.nii.gz"))
        lesions = []
        for lesion in lesion_image_paths:
            lesion_image = itk.array_from_image(itk.imread(lesion))
            lesions.append(lesion_image)
        lesion_masks = np.stack(lesions, axis=3) if lesions else [np.zeros_like(pet_image)]
        return pet_image, mri_image, lesion_masks

    def __getitem__(self, idx):
        record = self.records[idx]
        pet_volume, mri_volume, lesion_masks = self._get_volumes()

        return mri_volume, pet_volume, lesion_masks, record.labels

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
        loss_weights: Optional[Dict[str, float]] = None
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
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {
            "name": "step",
            "step_size": 100,
            "gamma": 0.96,
        }
        lw = loss_weights or {}
        self.seg_weight = float(lw.get("segmentation", 0.5))
        self.cls_weight = float(lw.get("classification", 1.0))

        self.scheduler_step_mode: Optional[str] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.scheduler_name: Optional[str] = None
        
        # Initialize loss function, for segmentation we let it be DICE
        self.criterion = DiceLoss()

        # Identify active tasks for Multi-Task Learning
        self.tasks = ["segmentation"]
        if self.cls_weight > 0:
            self.tasks.append("classification")
        
        self.multi_task_loss = MultiTaskLoss(self.tasks)

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
            records = load_patch_records(Path(data_csv))
        print("recordsssss", type(records))
        print(records)
        if not records:
            raise ValueError(f"No patch records found in {data_csv}")

        rng = np.random.default_rng(split_seed)
        series_ids = sorted({r.series_id for r in records})
        keep_count = max(1, int(round(len(series_ids) * train_ratio)))
        train_series = set(rng.choice(series_ids, size=keep_count, replace=False))
        train_records = [r for r in records if r.series_id in train_series]
        val_records = [r for r in records if r.series_id not in train_series]

        print("Train samples:")
        print(train_records, len(train_records))

        print("Val samples:")
        print(val_records, len(val_records))
        #For now we won't add data augmentation, but it will come as a TODO

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
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
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
    
    def train(
        self,
    ) -> dict:
        """Run the training loop using the loaders prepared beforehand."""
        # # Check if GPU available
        # if torch.cuda.is_available():
        #     print("GPU is on")
        #     self.device = "cuda"
        #     self.model = self.model.to("cuda")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            betas = (0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay)

        
        self.criterion = DiceLoss()
        
        cls_pos_weight = getattr(self, "pos_weight", torch.tensor(1.0))
        cls_pos_weight = cls_pos_weight.to(self.device)
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
        
        self._configure_scheduler()
        #Since we have to define our own training loop, we have to keep track
        #of these variables for best model, early stopping and learning rate decrease
        best_val_loss = float("inf")
        epochs_no_improve = 0
        lr_plateau_counter = 0

        #Also we want to keep track of our training history
        history = {
            "train_loss": [], 
            "train_acc": [], 
            "train_sensitivity": [], 
            "train_ppv": [], 
            "train_npv": [],
            "train_fpr": [],
            "train_specificity": [],
            "train_fnr": [],
            "val_loss": [], 
            "val_acc": [], 
            "val_sensitivity": [], 
            "val_ppv": [], 
            "val_npv": [],
            "train_dice": [], 
            "val_dice": [],
            "val_fpr": [],
            "val_specificity": [],
            "val_fnr": [],
        }

        eps = 1e-6
        for epoch in range(1, self.epochs + 1):
            # Get current MTL weights for display
            mtl_info = []
            for task, param in self.multi_task_loss.log_vars.items():
                weight = torch.exp(-param).item()
                mtl_info.append(f"{task}={weight:.4f}")
            mtl_str = ", ".join(mtl_info)
            print(f"Epoch {epoch}: , Weights: [{mtl_str}]")
            
            self.model.train()
            train_losses = []
            train_dice_scores = []
            train_mse_scores = [] 
            all_preds, all_labels = [], []
            train_seg_losses, train_cls_losses = [], []
            train_total_losses = []
            for x_batch, mask_batch, y_batch, coord_batch in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.float().unsqueeze(1)
                y_batch = y_batch.to(self.device, non_blocking=True)

                outputs = self.model(x_batch)
                # outputs[0] is segmentation, outputs[1] is classification (if any)
                # We only use segmentation output for the new loss
                seg_logits = outputs[0]
                seg_probs = torch.sigmoid(seg_logits)
                
                # Target for segmentation is the heatmap (mask_batch)
                # Assuming single channel output for now or matching channels
                targets = mask_batch[:, 0:1]
                
                seg_loss = self.criterion(seg_probs, targets)
                
                # Classification loss
                cls_loss = torch.tensor(0.0, device=self.device)
                if self.cls_weight > 0 and len(outputs) > 1:
                    cls_logits = outputs[1]
                    cls_loss = self.classification_loss(cls_logits, y_batch)
                
                # Combine losses using Multi-Task Uncertainty Weights
                loss_dict = {"segmentation": seg_loss}
                
                if self.cls_weight > 0 and len(outputs) > 1:
                    loss_dict["classification"] = cls_loss
                    
                loss = self.multi_task_loss(loss_dict)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                # Track individual task losses
                train_seg_losses.append(seg_loss.item())
                if "classification" in loss_dict:
                    train_cls_losses.append(cls_loss.item())
                train_total_losses.append(loss_dict.item())

                #Dice calculation (soft dice on soft targets)
                with torch.no_grad():
                    mask_probs = torch.sigmoid(outputs[0].detach().to(torch.float32))
                    targets = mask_batch[:, 0:1].detach().to(torch.float32)

                    pred_sum = mask_probs.sum(dim=(1, 2, 3, 4))
                    target_sum = targets.sum(dim=(1, 2, 3, 4))
                    intersection = (mask_probs * targets).sum(dim=(1, 2, 3, 4))
                    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
                    train_dice_scores.append(dice)

                #Aggregate metrics
                all_preds.append(outputs[1].detach())
                all_labels.append(y_batch.detach())
            
            #DICE loss stats
            train_dice_scores = torch.cat(train_dice_scores, dim=0).to(self.device, dtype=torch.float32)
            train_dice_mean = train_dice_scores.cpu().numpy().mean() if train_dice_scores.numel() > 0 else 0.0

            #Classification stats
            all_preds = torch.cat(all_preds, dim=0).to(self.device, dtype=torch.float32).contiguous()
            all_labels = torch.cat(all_labels, dim=0).to(self.device, dtype=torch.float32).contiguous()

            probs = torch.sigmoid(all_preds)
            train_preds_np = (probs >= 0.5).int().cpu().numpy().ravel()
            train_labels_np  = all_labels.int().cpu().numpy().ravel()

            tp = np.logical_and(train_preds_np == 1, train_labels_np == 1).sum()
            tn = np.logical_and(train_preds_np == 0, train_labels_np == 0).sum()
            fp = np.logical_and(train_preds_np == 1, train_labels_np == 0).sum()
            fn = np.logical_and(train_preds_np == 0, train_labels_np == 1).sum()
            train_acc = (tp + tn) / max(len(train_labels_np), 1)
            train_sens = tp / max(tp + fn, 1)          # sensitivity/recall
            train_ppv = tp / max(tp + fp, 1)           # precision/PPV
            train_npv = tn / max(tn + fn, 1)           # NPV
            train_fpr = fp / max(tn + fp, 1)           # false positive rate
            train_specificity = tn / max(tn + fp, 1)   # true negative rate
            train_fnr = fn / max(fn + tp, 1)           # false positive rate

            history["train_loss"].append(np.mean(train_losses))
            history["train_acc"].append(train_acc)
            history["train_sensitivity"].append(train_sens)
            history["train_ppv"].append(train_ppv)
            history["train_npv"].append(train_npv)
            history["train_fpr"].append(train_fpr)
            history["train_specificity"].append(train_specificity)
            history["train_fnr"].append(train_fnr)
            history["train_dice"].append(train_dice_mean)
            
            # Track individual task losses (use safe mean to avoid numpy warnings)
            train_seg_loss_mean = sum(train_seg_losses) / len(train_seg_losses) if train_seg_losses else 0.0
            train_cls_loss_mean = sum(train_cls_losses) / len(train_cls_losses) if train_cls_losses else 0.0
            train_total_loss_mean = sum(train_total_losses) / len(train_total_losses) if train_total_losses else 0.0
            history.setdefault("train_seg_loss", []).append(train_seg_loss_mean)
            history.setdefault("train_cls_loss", []).append(train_cls_loss_mean)
            history.setdefault("train_total_loss", []).append(train_total_loss_mean)

            # --- Validation ---
            self.model.eval()
            val_losses = []
            val_dice_scores = []
            val_preds_list, val_labels_list = [], []
            with torch.no_grad():
                val_losses = []
                val_dice_scores = []
                val_seg_losses, val_cls_losses = [], []
                val_total_losses = []
                for x_batch, mask_batch, y_batch, coord_batch in self.val_loader:
                    x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    mask_batch = mask_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    y_batch = y_batch.float().unsqueeze(1)
                    y_batch = y_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    coord_batch = coord_batch.to(self.device, dtype=torch.float32, non_blocking=True)
                    outputs = self.model(x_batch)
                    seg_logits = outputs[0]
                    seg_probs = torch.sigmoid(seg_logits)
                    targets = mask_batch[:, 0:1]
                    
                    seg_loss = self.criterion(seg_probs, targets)
                    
                    cls_loss = torch.tensor(0.0, device=self.device)
                    if self.cls_weight > 0 and len(outputs) > 1:
                        cls_logits = outputs[1]
                        cls_loss = self.classification_loss(cls_logits, y_batch)
                
                    loss_dict = {"segmentation": seg_loss}
                    if self.cls_weight > 0 and len(outputs) > 1:
                        loss_dict["classification"] = cls_loss
                    
                    loss = self.multi_task_loss(loss_dict)
                        
                    # Track individual task losses
                    val_seg_losses.append(seg_loss.item())
                    if "classification" in loss_dict:
                        val_cls_losses.append(cls_loss.item())
                    val_total_losses.append(loss.item())

                    #DICE loss (soft dice on soft targets)
                    mask_probs = torch.sigmoid(outputs[0].detach().to(torch.float32))
                    targets = mask_batch[:, 0:1].detach().to(torch.float32)

                    pred_sum = mask_probs.sum(dim=(1, 2, 3, 4))
                    target_sum = targets.sum(dim=(1, 2, 3, 4))
                    intersection = (mask_probs * targets).sum(dim=(1, 2, 3, 4))
                    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
                    val_dice_scores.append(dice)

                    #Classification loss
                    preds = outputs[1].detach().to(torch.float32)
                    labels = y_batch.detach().to(torch.float32)

                    # Sigmoid + threshold (done in float32 for numerical stability)
                    probs = torch.sigmoid(preds)
                    preds_np = (probs >= 0.5).int().cpu().numpy()
                    labs_np  = labels.int().cpu().numpy()
                    val_preds_list.append(preds_np)
                    val_labels_list.append(labs_np)

                    del preds, labels, probs, preds_np, labs_np, outputs, loss
                    torch.cuda.empty_cache()

            val_dice_scores = torch.cat(val_dice_scores, dim=0).to(self.device, dtype=torch.float32)
            val_dice_scores = self._safe_gather(val_dice_scores)
            val_dice_mean = val_dice_scores.cpu().numpy().mean() if val_dice_scores.numel() > 0 else 0.0
            
            val_preds_np = np.concatenate(val_preds_list, axis=0).ravel()
            val_labels_np = np.concatenate(val_labels_list, axis=0).ravel()

            tp = np.logical_and(val_preds_np == 1, val_labels_np == 1).sum()
            tn = np.logical_and(val_preds_np == 0, val_labels_np == 0).sum()
            fp = np.logical_and(val_preds_np == 1, val_labels_np == 0).sum()
            fn = np.logical_and(val_preds_np == 0, val_labels_np == 1).sum()
            val_acc = (tp + tn) / max(len(val_labels_np), 1)
            val_sens = tp / max(tp + fn, 1)
            val_ppv = tp / max(tp + fp, 1)
            val_npv = tn / max(tn + fn, 1)
            val_fpr = fp / max(tn + fp, 1)         
            val_specificity = tn / max(tn + fp, 1)   
            val_fnr = fn / max(fn + tp, 1)
            val_loss = np.mean(val_losses)
            
            # Calculate mean task losses (use safe mean to avoid numpy warnings)
            val_seg_loss_mean = sum(val_seg_losses) / len(val_seg_losses) if val_seg_losses else 0.0
            val_cls_loss_mean = sum(val_cls_losses) / len(val_cls_losses) if val_cls_losses else 0.0
            val_total_loss_mean = sum(val_total_losses) / len(val_total_losses) if val_total_losses else 0.0
            # Use segmentation loss as primary validation metric
            val_loss = val_seg_loss_mean
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_sensitivity"].append(val_sens)
            history["val_ppv"].append(val_ppv)
            history["val_npv"].append(val_npv)
            history["val_fpr"].append(val_fpr)
            history["val_specificity"].append(val_specificity)
            history["val_fnr"].append(val_fnr)
            history["val_dice"].append(val_dice_mean)
            history.setdefault("val_seg_loss", []).append(val_seg_loss_mean)
            history.setdefault("val_cls_loss", []).append(val_cls_loss_mean)
            history.setdefault("val_total_loss", []).append(val_total_loss_mean)
            
            # Log Multi-Task Weights (sigma^2 = exp(s))
            mtl_weights = {}
            for task, param in self.multi_task_loss.log_vars.items():
                sigma2 = torch.exp(param).item()
                weight = 1.0 / sigma2
                mtl_weights[f"mtl_weight/{task}"] = weight
                history.setdefault(f"mtl_weight_{task}", []).append(weight)

            train_seg_loss_mean = history["train_seg_loss"][-1]
            train_cls_loss_mean = history["train_cls_loss"][-1]
            train_total_loss_mean = history["train_total_loss"][-1]
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Create table header with dynamic variables
            print(f"\n{'='*150}")
            print(f"Epoch {epoch}/{self.epochs} | LR: {current_lr:.2e} | Weights: [{mtl_str}]")
            print(f"{'='*150}")
            
            # Metrics table
            print(f"{'':8} | {'TotalLoss':>9} | {'SegLoss':>8} | {'ClsLoss':>8} | {'Acc':>6} | {'Sens':>6} | {'FPR':>6} | {'Spec':>6} | {'FNR':>6} | {'PPV':>6} | {'NPV':>6} | {'Dice':>6} ")
            print(f"{'-'*150}")
            print(
                f"{'Train':8} | {train_total_loss_mean:9.4f} | {train_seg_loss_mean:8.4f} | {train_cls_loss_mean:8.4f} | "
                f"{train_acc:6.4f} | {train_sens:6.4f} | {train_fpr:6.4f} | {train_specificity:6.4f} | {train_fnr:6.4f} | {train_ppv:6.4f} | {train_npv:6.4f} | "
                f"{train_dice_mean:6.4f}"
            )
            print(
                f"{'Val':8} | {val_total_loss_mean:9.4f} | {val_seg_loss_mean:8.4f} | {val_cls_loss_mean:8.4f} | "
                f"{val_acc:6.4f} | {val_sens:6.4f} | {val_fpr:6.4f} | {val_specificity:6.4f} | {val_fnr:6.4f} | {val_ppv:6.4f} | {val_npv:6.4f} | "
                f"{val_dice_mean:6.4f}"
            )
            print(f"{'='*150}\n")
            
            # per-epoch checkpoint (optional)
            ckpt_path = Path(self.checkpoint_path)
            epoch_ckpt = ckpt_path.with_name(f"{epoch}_{ckpt_path.name}")
            epoch_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), epoch_ckpt)
            # --- Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Validation loss improved: saved to {self.checkpoint_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1


            # --- Early stopping ---
            if epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            # --- ReduceLROnPlateau (manual) ---
            if self.scheduler_name == "step" and len(history["val_loss"]) > self.lr_reduction_patience:
                if lr_plateau_counter > self.lr_reduction_patience and history["val_loss"][-1] >= history["val_loss"][-(self.lr_reduction_patience + 1)]:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = max(old_lr * self.lr_reduction_factor, self.min_lr)
                    if new_lr < old_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"ReduceLROnPlateau: lr reduced from {old_lr:.1e} to {new_lr:.1e}")
                        lr_plateau_counter = 0
                else:
                    lr_plateau_counter += 1

            if self.scheduler:
                if self.scheduler_step_mode == "epoch":
                    self.scheduler.step()
                elif self.scheduler_step_mode == "plateau":
                    self.scheduler.step(history["val_loss"][-1])

        return history