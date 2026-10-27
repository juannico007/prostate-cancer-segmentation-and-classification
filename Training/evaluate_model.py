import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import itk
import torch

from model import UNet
from training_pipeline import (
    CLASSIFICATION_TARGETS,
    ProstateDataset,
    collate_fn,
    load_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG = REPO_ROOT / "Training/run_config.json"
SEGMENTATION_LABELS = ("background", "score_3", "score_4", "score_5")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate patient classification and four-class segmentation; save case visualizations."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUN_CONFIG,
    )
    parser.add_argument("--history", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("val", "train", "test", "all"))
    parser.add_argument("--num-examples", type=int)
    return parser.parse_args()


def decode_severity(logits, classification_loss="cross_entropy"):
    if classification_loss == "coral":
        return int((logits.float() > 0).sum().item())
    return int(logits.float().argmax(dim=-1).item())


def severity_probabilities(logits, classification_loss="cross_entropy"):
    logits = logits.float()
    if classification_loss != "coral":
        return torch.softmax(logits, dim=-1)
    cumulative_probabilities = torch.sigmoid(logits)
    cumulative_probabilities = torch.cummin(cumulative_probabilities, dim=-1).values
    return torch.cat(
        (
            1 - cumulative_probabilities[..., :1],
            cumulative_probabilities[..., :-1] - cumulative_probabilities[..., 1:],
            cumulative_probabilities[..., -1:],
        ),
        dim=-1,
    )


def plot_confusion_matrix(matrix, output_path, title, labels, axis_label):
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Count")
    positions = np.arange(len(labels))
    axis.set_xticks(positions, labels=labels, rotation=35, ha="right")
    axis.set_yticks(positions, labels=labels)
    axis.set_xlabel(f"Predicted {axis_label}")
    axis.set_ylabel(f"Ground-truth {axis_label}")
    axis.set_title(title)
    midpoint = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_loss_curves(history_path, output_path):
    with history_path.open("rb") as stream:
        history = pickle.load(stream)

    loss_groups = (
        ("total", "Total loss"),
        ("segmentation", "Segmentation loss"),
        ("severity", "Patient score loss"),
    )
    figure, axes = plt.subplots(1, len(loss_groups), figsize=(15, 4.5), constrained_layout=True)
    for axis, (key, title) in zip(axes, loss_groups):
        plotted = False
        for split, label in (("train", "Train"), ("val", "Validation")):
            values = history.get(f"{split}_{key}")
            if values:
                axis.plot(np.arange(1, len(values) + 1), values, label=label)
                plotted = True
        if not plotted:
            raise ValueError(f"No train/validation history found for loss '{key}' in {history_path}")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_segmentation_channels(prediction, mri_path, output_dir, patient_id):
    reference_image = itk.imread(mri_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for class_index, class_name in enumerate(SEGMENTATION_LABELS):
        binary_mask = (prediction == class_index).astype(np.uint8)
        mask_image = itk.image_from_array(binary_mask)
        mask_image.CopyInformation(reference_image)
        output_path = output_dir / f"patient_{patient_id}_{class_name}.nii.gz"
        itk.imwrite(mask_image, output_path, compression=True)
        saved_paths.append(output_path)
    return saved_paths


def segmentation_metrics(confusion):
    metrics = {}
    for class_index, label in enumerate(SEGMENTATION_LABELS):
        true_positive = int(confusion[class_index, class_index])
        ground_truth_count = int(confusion[class_index, :].sum())
        predicted_count = int(confusion[:, class_index].sum())
        denominator = ground_truth_count + predicted_count
        metrics[label] = {
            "support": ground_truth_count,
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / ground_truth_count if ground_truth_count else 0.0,
            "dice": 2 * true_positive / denominator if denominator else None,
        }
    foreground_dice = [
        metrics[label]["dice"] for label in SEGMENTATION_LABELS[1:]
        if metrics[label]["dice"] is not None
    ]
    metrics["mean_foreground_dice"] = float(np.mean(foreground_dice)) if foreground_dice else None
    return metrics


def main():
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)

    def resolve_path(path):
        path = Path(path)
        return (path if path.is_absolute() else REPO_ROOT / path).resolve()

    experiment_dir = resolve_path(config["experiment_dir"])
    artifact_id = f"{config['model_name']}_{config['model_version']}"
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else (
        experiment_dir / f"{artifact_id}.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    data_dir = resolve_path(args.data_dir or config["data_dir"])
    data_csv = resolve_path(args.data_csv or config["data_csv"])
    output_dir = resolve_path(args.output_dir) if args.output_dir else experiment_dir / "evaluation" / config["split"]
    output_dir.mkdir(parents=True, exist_ok=True)
    segmentations_dir = output_dir / "predicted_segmentations"
    history_path = resolve_path(args.history) if args.history else (
        experiment_dir / f"{artifact_id}_history.pickle"
    )
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    loss_plot_path = output_dir / "loss_curves.png"
    plot_loss_curves(history_path, loss_plot_path)

    selected_split = args.split or config.get("split", "test")
    split_dir = experiment_dir / "splits"
    split_paths = {
        "train": split_dir / "train_split.csv",
        "validation": split_dir / "validation_split.csv",
        "test": split_dir / "test_split.csv",
    }
    missing_splits = [path for path in split_paths.values() if not path.is_file()]
    if missing_splits:
        raise FileNotFoundError(
            "Training split manifests not found; run training with this config first: "
            + ", ".join(str(path) for path in missing_splits)
        )
    splits = {name: load_records(path) for name, path in split_paths.items()}
    if selected_split == "all":
        records = [record for split_records_list in splits.values() for record in split_records_list]
    else:
        split_name = "validation" if selected_split == "val" else selected_split
        if split_name not in splits:
            raise ValueError(f"Unknown configured split: {selected_split}")
        records = splits[split_name]
    if not records:
        raise ValueError(f"No records selected for split '{selected_split}'.")
    num_examples = args.num_examples if args.num_examples is not None else int(config.get("num_examples", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_loss = config.get("classification_loss", "cross_entropy")
    classification_target = config.get("classification_target", "primary_grade")
    if classification_target not in CLASSIFICATION_TARGETS:
        raise ValueError(
            f"classification_target must be one of {', '.join(CLASSIFICATION_TARGETS)}."
        )
    model_kwargs = dict(config["unet"])
    model_kwargs["ordinal_classification"] = classification_loss == "coral"
    if classification_target == "binary_primary_3" and model_kwargs["num_severity_levels"] != 2:
        raise ValueError("binary_primary_3 classification requires num_severity_levels=2.")
    dataset = ProstateDataset(
        data_dir,
        records,
        classification_target=classification_target,
    )
    model = UNet(**model_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    use_amp = device.type == "cuda" and bool(config.get("mixed_precision", True))

    classification_levels = model.num_severity_levels
    classification_confusion = np.zeros(
        (classification_levels, classification_levels),
        dtype=np.int64,
    )
    segmentation_confusion = np.zeros((model.out_channels, model.out_channels), dtype=np.int64)
    patient_rows = []
    classification_errors = []
    evaluated_patients = 0
    segmentation_patients = 0
    skipped_patients = 0
    example_count = 0

    for index, record in enumerate(records):
        if not dataset.has_required_volumes(record):
            print(f"[skip] patient {record.series_id}: missing PET or MRI_T2 volume")
            skipped_patients += 1
            continue
        try:
            sample = dataset[index]
            batch = collate_fn([sample])
        except (FileNotFoundError, ValueError) as error:
            print(f"[skip] patient {record.series_id}: {error}")
            skipped_patients += 1
            continue

        image = batch["image"].to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(image)
        predicted_class = decode_severity(
            output["severity_logits"][0],
            classification_loss,
        )
        class_probabilities = severity_probabilities(
            output["severity_logits"][0],
            classification_loss,
        ).cpu().tolist()
        true_score = int(batch["patient_score"][0])
        true_class = int(batch["classification_target"][0])
        classification_confusion[true_class, predicted_class] += 1
        classification_errors.append(abs(true_class - predicted_class))
        evaluated_patients += 1

        segmentation_valid = bool(batch["segmentation_valid"][0])
        patient_row = {
            "patient_id": record.series_id,
            "true_max_score": true_score,
            "true_classification_label": true_class,
            "predicted_classification_label": predicted_class,
            "absolute_classification_error": abs(true_class - predicted_class),
            "segmentation_evaluated": segmentation_valid,
        }
        patient_row.update({
            f"probability_class_{class_index}": probability
            for class_index, probability in enumerate(class_probabilities)
        })
        if classification_target == "primary_grade":
            patient_row["predicted_max_score"] = predicted_class + 1
        patient_rows.append(patient_row)
        probability_summary = ", ".join(
            f"{class_index}: {probability:.3f}"
            for class_index, probability in enumerate(class_probabilities)
        )
        print(
            f"[evaluation] patient {record.series_id}: max PRIMARY={true_score}, "
            f"class={true_class}, predicted class={predicted_class}, "
            f"class probabilities=[{probability_summary}]"
        )

        prediction = output["segmentation_logits"].argmax(dim=1)[0].cpu().numpy()
        if segmentation_valid:
            target = batch["segmentation"][0].numpy()
            encoded_pairs = target.reshape(-1) * model.out_channels + prediction.reshape(-1)
            segmentation_confusion += np.bincount(
                encoded_pairs,
                minlength=model.out_channels * model.out_channels,
            ).reshape(model.out_channels, model.out_channels)
            segmentation_patients += 1
        else:
            print(f"[info] patient {record.series_id}: classification evaluated; incomplete score 3-5 masks excluded from segmentation metrics")

        if example_count < num_examples:
            patient_path = data_dir / str(record.series_id)
            mri_paths = list(patient_path.rglob("*MRI_T2.nii.gz"))
            if len(mri_paths) == 1:
                save_segmentation_channels(
                    prediction,
                    mri_paths[0],
                    segmentations_dir,
                    record.series_id,
                )
                example_count += 1

        del image, output

    if not evaluated_patients:
        raise RuntimeError("No patients with readable MRI and PET volumes were available for evaluation.")

    classification_errors = np.asarray(classification_errors)
    if classification_target == "binary_primary_3":
        classification_label_names = ["below_primary_3", "primary_3_to_5"]
    else:
        classification_label_names = [
            str(score) for score in range(1, classification_levels + 1)
        ]
    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": selected_split,
        "split_seed": int(config.get("split_seed", 42)),
        "split_patient_counts": {name: len(records) for name, records in splits.items()},
        "evaluated_patients": evaluated_patients,
        "segmentation_patients": segmentation_patients,
        "skipped_patients": skipped_patients,
        "classification_target": classification_target,
        "patient_classification": {
            "accuracy": float(
                np.trace(classification_confusion) / classification_confusion.sum()
            ),
            "mean_absolute_error": float(classification_errors.mean()),
            "confusion_matrix_labels": (
                list(range(classification_levels))
                if classification_target == "binary_primary_3"
                else list(range(1, classification_levels + 1))
            ),
            "confusion_matrix_label_names": classification_label_names,
            "confusion_matrix": classification_confusion.tolist(),
        },
        "segmentation": {
            "class_labels": list(SEGMENTATION_LABELS),
            "confusion_matrix": segmentation_confusion.tolist(),
            "classes": segmentation_metrics(segmentation_confusion),
        },
    }
    if classification_target == "primary_grade":
        metrics["patient_classification"]["within_one_grade"] = float(
            np.mean(classification_errors <= 1)
        )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    predictions_path = output_dir / "patient_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=patient_rows[0].keys())
        writer.writeheader()
        writer.writerows(patient_rows)

    plot_labels = (
        ["0: below PRIMARY 3", "1: PRIMARY 3-5"]
        if classification_target == "binary_primary_3"
        else classification_label_names
    )
    plot_confusion_matrix(
        classification_confusion,
        output_dir / "patient_classification_confusion_matrix.png",
        "Patient classification",
        plot_labels,
        "class",
    )
    plot_confusion_matrix(
        segmentation_confusion,
        output_dir / "segmentation_confusion_matrix.png",
        "Four-class voxel segmentation",
        SEGMENTATION_LABELS,
        "segmentation class",
    )

    print(f"Evaluated {evaluated_patients} patients; segmentation evaluated on {segmentation_patients}; skipped {skipped_patients}.")
    print(
        "Patient classification accuracy: "
        f"{metrics['patient_classification']['accuracy']:.3f}"
    )
    print(
        "Patient classification MAE: "
        f"{metrics['patient_classification']['mean_absolute_error']:.3f}"
    )
    print(f"Mean foreground segmentation Dice: {metrics['segmentation']['classes']['mean_foreground_dice']}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved patient predictions: {predictions_path}")
    print(f"Loaded split metadata CSVs from: {split_dir}")
    print(f"Saved loss curves: {loss_plot_path}")
    print(f"Saved per-channel segmentations: {segmentations_dir}")
    print(f"Saved confusion matrices: {output_dir}")


if __name__ == "__main__":
    main()
