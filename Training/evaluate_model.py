import argparse
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
from training_pipeline import ProstateDataset, collate_fn, load_records


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG = REPO_ROOT / "Training/run_config.json"
SEGMENTATION_LABELS = ("background", "score_3", "score_4", "score_5")
BINARY_SEGMENTATION_LABELS = ("background", "lesion")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate four-class segmentation and save case visualizations."
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
    )
    figure, axes = plt.subplots(1, len(loss_groups), figsize=(10, 4.5), constrained_layout=True)
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


def save_segmentation_channels(
    prediction,
    mri_path,
    output_dir,
    patient_id,
    labels=SEGMENTATION_LABELS,
):
    reference_image = itk.imread(mri_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for class_index, class_name in enumerate(labels):
        binary_mask = (prediction == class_index).astype(np.uint8)
        mask_image = itk.image_from_array(binary_mask)
        mask_image.CopyInformation(reference_image)
        output_path = output_dir / f"patient_{patient_id}_{class_name}.nii.gz"
        itk.imwrite(mask_image, output_path, compression=True)
        saved_paths.append(output_path)
    return saved_paths


def segmentation_metrics(confusion, labels=SEGMENTATION_LABELS):
    metrics = {}
    for class_index, label in enumerate(labels):
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
        metrics[label]["dice"] for label in labels[1:]
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
    grouped_normalization = config.get("grouped_normalization", True)
    if not isinstance(grouped_normalization, bool):
        raise ValueError("grouped_normalization must be a boolean in the config file.")
    binary_segmentation = config.get("binary_segmentation", False)
    if not isinstance(binary_segmentation, bool):
        raise ValueError("binary_segmentation must be a boolean in the config file.")

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
    unet_kwargs = dict(config["unet"])
    if binary_segmentation:
        unet_kwargs["out_channels"] = 1
    model = UNet(**unet_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        normalization_stats = (
            checkpoint.get("normalization_stats") if grouped_normalization else None
        )
        if grouped_normalization and normalization_stats is None:
            raise ValueError(
                f"Checkpoint is missing normalization parameters: {checkpoint_path}"
            )
    else:
        state = checkpoint
        normalization_stats = None
        print(
            "[Warning] Legacy checkpoint has no normalization parameters; "
            "evaluating without input normalization."
        )
    model.load_state_dict(state)
    model.eval()
    use_amp = device.type == "cuda" and bool(config.get("mixed_precision", True))
    dataset = ProstateDataset(
        data_dir,
        records,
        normalization_stats=normalization_stats,
        binary_segmentation=binary_segmentation,
    )
    if normalization_stats is not None:
        print(f"Using training-set normalization parameters: {normalization_stats}")
    elif not grouped_normalization:
        print("Grouped training-set normalization is disabled.")

    segmentation_labels = (
        BINARY_SEGMENTATION_LABELS if binary_segmentation else SEGMENTATION_LABELS
    )
    expected_output_channels = 1 if binary_segmentation else len(segmentation_labels)
    if model.out_channels != expected_output_channels:
        raise ValueError(
            f"Model has {model.out_channels} output channels, but the selected "
            f"segmentation mode requires {expected_output_channels}."
        )
    segmentation_classes = len(segmentation_labels)
    segmentation_confusion = np.zeros(
        (segmentation_classes, segmentation_classes),
        dtype=np.int64,
    )
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
        evaluated_patients += 1

        segmentation_valid = bool(batch["segmentation_valid"][0])
        if binary_segmentation:
            prediction = (
                torch.sigmoid(output["segmentation_logits"])[0, 0] >= 0.5
            ).to(torch.int64).cpu().numpy()
        else:
            prediction = output["segmentation_logits"].argmax(dim=1)[0].cpu().numpy()

        if segmentation_valid:
            target = batch["segmentation"][0].numpy()
            if binary_segmentation:
                target = target[0]
            target = target.astype(np.int64, copy=False)
            encoded_pairs = (
                target.reshape(-1) * segmentation_classes + prediction.reshape(-1)
            )
            segmentation_confusion += np.bincount(
                encoded_pairs,
                minlength=segmentation_classes * segmentation_classes,
            ).reshape(segmentation_classes, segmentation_classes)
            segmentation_patients += 1
        else:
            print(f"[info] patient {record.series_id}: incomplete score 3-5 masks excluded from segmentation metrics")

        if example_count < num_examples:
            patient_path = data_dir / str(record.series_id)
            mri_paths = list(patient_path.rglob("*MRI_T2.nii.gz"))
            if len(mri_paths) == 1:
                save_segmentation_channels(
                    prediction,
                    mri_paths[0],
                    segmentations_dir,
                    record.series_id,
                    segmentation_labels,
                )
                example_count += 1

        del image, output

    if not evaluated_patients:
        raise RuntimeError("No patients with readable MRI and PET volumes were available for evaluation.")

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": selected_split,
        "split_seed": int(config.get("split_seed", 42)),
        "split_patient_counts": {name: len(records) for name, records in splits.items()},
        "evaluated_patients": evaluated_patients,
        "segmentation_patients": segmentation_patients,
        "skipped_patients": skipped_patients,
        "segmentation": {
            "class_labels": list(segmentation_labels),
            "confusion_matrix": segmentation_confusion.tolist(),
            "classes": segmentation_metrics(segmentation_confusion, segmentation_labels),
        },
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    plot_confusion_matrix(
        segmentation_confusion,
        output_dir / "segmentation_confusion_matrix.png",
        "Binary voxel segmentation" if binary_segmentation else "Four-class voxel segmentation",
        segmentation_labels,
        "segmentation class",
    )

    print(f"Evaluated {evaluated_patients} patients; segmentation evaluated on {segmentation_patients}; skipped {skipped_patients}.")
    print(f"Mean foreground segmentation Dice: {metrics['segmentation']['classes']['mean_foreground_dice']}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Loaded split metadata CSVs from: {split_dir}")
    print(f"Saved loss curves: {loss_plot_path}")
    print(f"Saved per-class segmentations: {segmentations_dir}")
    print(f"Saved confusion matrices: {output_dir}")


if __name__ == "__main__":
    main()
