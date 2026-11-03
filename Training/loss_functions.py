import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_square_class_weights(class_counts, power=2.0):
    if power <= 0:
        raise ValueError("class_weight_power must be greater than zero.")
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    present = counts > 0
    if not present.any():
        raise ValueError("Cannot compute class weights when all class counts are zero.")
    frequencies = counts[present] / counts[present].sum()
    weights = torch.zeros_like(counts)
    weights[present] = frequencies.pow(-power)
    weights[present] /= weights[present].mean()
    return weights.float()


class TrainingLosses(nn.Module):
    def __init__(
        self,
        segmentation_classes,
        loss_weights=None,
        class_weight_power=2.0,
    ):
        super().__init__()
        if class_weight_power is not None and class_weight_power <= 0:
            raise ValueError("class_weight_power must be greater than zero or None.")

        loss_weights = loss_weights or {}
        self.segmentation_classes = segmentation_classes
        self.segmentation_weight = float(loss_weights.get("segmentation", 1.0))
        self.class_weight_power = (
            float(class_weight_power) if class_weight_power is not None else None
        )
        self.register_buffer("segmentation_class_weights", None)

    def configure_class_weights(self, segmentation_counts):
        if self.class_weight_power is None:
            self.segmentation_class_weights = None
            return

        self.segmentation_class_weights = inverse_square_class_weights(
            segmentation_counts,
            self.class_weight_power,
        )

    def _segmentation_loss(self, logits, targets, valid):
        if not valid.any():
            zero_loss = logits.sum() * 0
            return zero_loss, zero_loss.detach()

        valid_logits = logits[valid]
        valid_targets = targets[valid]
        cross_entropy = F.cross_entropy(
            valid_logits,
            valid_targets,
            weight=self.segmentation_class_weights,
        )
        probabilities = torch.softmax(valid_logits, dim=1)
        dice_scores = []
        for class_index in range(1, self.segmentation_classes):
            class_targets = valid_targets == class_index
            if class_targets.any():
                class_probabilities = probabilities[:, class_index]
                intersection = (class_probabilities * class_targets).sum()
                denominator = class_probabilities.sum() + class_targets.sum()
                dice_scores.append((2 * intersection + 1e-5) / (denominator + 1e-5))
        dice = torch.stack(dice_scores).mean() if dice_scores else cross_entropy.new_tensor(1.0)
        return cross_entropy + (1 - dice), dice

    def forward(self, outputs, batch):
        segmentation_logits = outputs["segmentation_logits"].float()
        segmentation_targets = batch["segmentation"].to(
            segmentation_logits.device,
            non_blocking=True,
        )
        segmentation_valid = batch["segmentation_valid"].to(
            segmentation_logits.device,
            non_blocking=True,
        )
        segmentation_loss, dice = self._segmentation_loss(
            segmentation_logits,
            segmentation_targets,
            segmentation_valid,
        )
        total_loss = self.segmentation_weight * segmentation_loss
        return {
            "total": total_loss,
            "segmentation": segmentation_loss,
            "dice": dice,
        }