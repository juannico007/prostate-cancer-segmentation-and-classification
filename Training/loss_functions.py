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
        severity_levels,
        loss_weights=None,
        classification_loss="cross_entropy",
        gaussian_sigma=1.0,
        class_weight_power=2.0,
    ):
        super().__init__()
        if classification_loss not in {"cross_entropy", "coral", "gaussian"}:
            raise ValueError(
                "classification_loss must be 'cross_entropy', 'coral', or 'gaussian'."
            )
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be greater than zero.")
        if class_weight_power is not None and class_weight_power <= 0:
            raise ValueError("class_weight_power must be greater than zero or None.")

        loss_weights = loss_weights or {}
        self.segmentation_classes = segmentation_classes
        self.severity_levels = severity_levels
        self.segmentation_weight = float(loss_weights.get("segmentation", 1.0))
        self.classification_weight = float(loss_weights.get("classification", 1.0))
        self.classification_loss = classification_loss
        self.gaussian_sigma = float(gaussian_sigma)
        self.class_weight_power = (
            float(class_weight_power) if class_weight_power is not None else None
        )
        self.register_buffer("segmentation_class_weights", None)
        self.register_buffer("severity_class_weights", None)

    def configure_class_weights(self, segmentation_counts, severity_counts):
        if self.class_weight_power is None:
            self.segmentation_class_weights = None
            self.severity_class_weights = None
            return

        self.segmentation_class_weights = inverse_square_class_weights(
            segmentation_counts,
            self.class_weight_power,
        )
        self.severity_class_weights = inverse_square_class_weights(
            severity_counts,
            self.class_weight_power,
        )

    @staticmethod
    def _weighted_mean(losses, weights):
        if weights is None:
            return losses.mean()
        return (losses * weights).sum() / weights.sum().clamp_min(
            torch.finfo(losses.dtype).eps
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

    def _classification_loss(self, logits, patient_scores):
        targets = patient_scores - 1
        sample_weights = (
            self.severity_class_weights[targets]
            if self.severity_class_weights is not None
            else None
        )

        if self.classification_loss == "coral":
            threshold_count = self.severity_levels - 1
            thresholds = torch.arange(threshold_count, device=logits.device)
            ordinal_targets = (targets.unsqueeze(1) > thresholds).float()
            if logits.shape[1] != threshold_count:
                raise ValueError(
                    f"CORAL expects {threshold_count} threshold logits, "
                    f"received {logits.shape[1]}."
                )
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits,
                ordinal_targets,
                reduction="none",
            ).mean(dim=1)
            loss = self._weighted_mean(per_sample_loss, sample_weights)
            predictions = (logits > 0).sum(dim=1) + 1
        elif self.classification_loss == "gaussian":
            class_indices = torch.arange(
                self.severity_levels,
                device=logits.device,
                dtype=logits.dtype,
            )
            gaussian_targets = torch.exp(
                -0.5
                * ((class_indices.unsqueeze(0) - targets.unsqueeze(1))
                   / self.gaussian_sigma) ** 2
            )
            gaussian_targets = gaussian_targets / gaussian_targets.sum(
                dim=1,
                keepdim=True,
            )
            per_sample_loss = -(
                gaussian_targets * F.log_softmax(logits, dim=1)
            ).sum(dim=1)
            loss = self._weighted_mean(per_sample_loss, sample_weights)
            predictions = logits.argmax(dim=1) + 1
        else:
            loss = F.cross_entropy(
                logits,
                targets,
                weight=self.severity_class_weights,
            )
            predictions = logits.argmax(dim=1) + 1

        return loss, predictions

    def forward(self, outputs, batch):
        segmentation_logits = outputs["segmentation_logits"].float()
        severity_logits = outputs["severity_logits"].float()
        segmentation_targets = batch["segmentation"].to(
            segmentation_logits.device,
            non_blocking=True,
        )
        segmentation_valid = batch["segmentation_valid"].to(
            segmentation_logits.device,
            non_blocking=True,
        )
        patient_scores = batch["patient_score"].to(
            severity_logits.device,
            non_blocking=True,
        )

        segmentation_loss, dice = self._segmentation_loss(
            segmentation_logits,
            segmentation_targets,
            segmentation_valid,
        )
        severity_loss, severity_predictions = self._classification_loss(
            severity_logits,
            patient_scores,
        )
        severity_mae = (severity_predictions - patient_scores).abs().float().mean()
        severity_accuracy = (severity_predictions == patient_scores).float().mean()
        total_loss = (
            self.segmentation_weight * segmentation_loss
            + self.classification_weight * severity_loss
        )
        return {
            "total": total_loss,
            "segmentation": segmentation_loss,
            "severity": severity_loss,
            "dice": dice,
            "severity_mae": severity_mae,
            "severity_accuracy": severity_accuracy,
        }