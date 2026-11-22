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
from training_pipeline import (
    ProstateDataset,
    collate_fn,
    load_records,
    segmentation_labels as get_segmentation_labels,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG = REPO_ROOT / "Training/run_config.json"
SEGMENTATION_LABELS = ("background", "score_1", "score_2", "score_3", "score_4", "score_5")
BINARY_SEGMENTATION_LABELS = ("background", "lesion")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation and save case visualizations."
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
    parser.add_argument("--num-examples", type=int, help="Patients exported for visualization.")
    parser.add_argument(
        "--example-selection",
        choices=EXAMPLE_SELECTIONS,
        default="spread",
        help="Which patients to export: spread over the binary foreground Dice range (default), worst, best, or first.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="With k_folds > 1 in the config, evaluate the model of this 1-based fold.",
    )
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
        ("segmentation", "Segmentation loss", "Loss"),
        ("dice", "Soft Dice", "Dice"),
    )
    figure, axes = plt.subplots(1, len(loss_groups), figsize=(10, 4.5), constrained_layout=True)
    for axis, (key, title, y_label) in zip(axes, loss_groups):
        plotted = False
        for split, label in (("train", "Train"), ("val", "Validation")):
            values = history.get(f"{split}_{key}")
            if values:
                axis.plot(np.arange(1, len(values) + 1), values, label=label)
                plotted = True
        if not plotted:
            if key == "dice":
                axis.set_visible(False)
                continue
            raise ValueError(f"No train/validation history found for loss '{key}' in {history_path}")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


EXAMPLE_SELECTIONS = ("spread", "worst", "best", "first")


def save_visualization_case(output_dir, patient_path, prediction, probability_maps, target):
    """Write prediction, evaluation target, and probability maps on the MRI grid."""
    mri_paths = list(patient_path.rglob("*MRI_T2.nii.gz"))
    if len(mri_paths) != 1:
        raise ValueError(
            f"Visualization export requires exactly one MRI_T2 for {patient_path}; "
            f"found {len(mri_paths)}."
        )
    reference_image = itk.imread(mri_paths[0])
    arrays = [
        (prediction.astype(np.uint8), "prediction.nii.gz"),
        (target.astype(np.uint8), "target.nii.gz"),
        *[(probability.astype(np.float32), name) for name, probability in probability_maps.items()],
    ]
    expected_shape = tuple(reversed(tuple(itk.size(reference_image))))
    for array, name in arrays:
        if array.shape != expected_shape:
            raise ValueError(
                f"Cannot export {name} for {patient_path}: shape {array.shape} "
                f"does not match MRI shape {expected_shape}."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for array, name in arrays:
        image = itk.image_from_array(np.ascontiguousarray(array))
        image.CopyInformation(reference_image)
        itk.imwrite(image, output_dir / name, compression=True)
    return True


def select_example_patients(per_patient_dice, evaluated_ids, num_examples, selection):
    """Choose which patients to export, ordered by binary foreground Dice where available."""
    if num_examples <= 0:
        return []
    if selection == "first":
        return list(evaluated_ids[:num_examples])
    ranked = sorted(
        (patient for patient, dice in per_patient_dice.items() if dice is not None),
        key=lambda patient: per_patient_dice[patient],
    )
    if selection == "best":
        return ranked[::-1][:num_examples]
    if selection == "worst":
        return ranked[:num_examples]
    if len(ranked) <= num_examples:
        return ranked
    # Evenly spaced ranks always include the worst and best patient.
    positions = np.linspace(0, len(ranked) - 1, num_examples).round().astype(int)
    return [ranked[position] for position in dict.fromkeys(positions.tolist())]


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
    foreground_true_positive = int(confusion[1:, 1:].sum())
    foreground_denominator = int(confusion[1:, :].sum() + confusion[:, 1:].sum())
    metrics["foreground_dice"] = (
        2 * foreground_true_positive / foreground_denominator
        if foreground_denominator else None
    )
    return metrics


def resolve_path(path):
    path = Path(path)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_config(config_path):
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    for key, default in (
        ("grouped_normalization", True),
        ("binary_segmentation", False),
        ("exclude_low_score_only_patients", True),
    ):
        if not isinstance(config.get(key, default), bool):
            raise ValueError(f"{key} must be a boolean in the config file.")
    return config


def config_k_folds(config):
    k_folds = config.get("k_folds")
    return k_folds if isinstance(k_folds, int) and not isinstance(k_folds, bool) and k_folds > 1 else None


def load_split_records(run_dir):
    split_dir = run_dir / "splits"
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
    return {name: load_records(path) for name, path in split_paths.items()}


def select_records(splits, selected_split):
    if selected_split == "all":
        return [record for split_records_list in splits.values() for record in split_records_list]
    split_name = "validation" if selected_split == "val" else selected_split
    if split_name not in splits:
        raise ValueError(f"Unknown configured split: {selected_split}")
    return splits[split_name]


def load_model(config, checkpoint_path, device):
    """Build the configured UNet and load a checkpoint; returns (model, normalization_stats)."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    grouped_normalization = config.get("grouped_normalization", True)
    unet_kwargs = dict(config["unet"])
    if config.get("binary_segmentation", False):
        unet_kwargs["out_channels"] = 1
    elif config.get("exclude_low_score_only_patients", True):
        unet_kwargs["out_channels"] = 5
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
    return model, normalization_stats


def evaluate_records(
    model,
    records,
    config,
    data_dir,
    normalization_stats,
    device,
    segmentations_dir=None,
    num_examples=0,
    example_selection="spread",
    per_class_probability_maps=False,
):
    """Run inference on records, accumulate voxel confusion and per-patient Dice, and export examples.

    Examples are chosen after all patients are scored (see select_example_patients) and
    written to segmentations_dir/patient_<id>/.
    """
    binary_segmentation = config.get("binary_segmentation", False)
    use_amp = device.type == "cuda" and bool(config.get("mixed_precision", True))
    dataset = ProstateDataset(
        data_dir,
        records,
        normalization_stats=normalization_stats,
        binary_segmentation=binary_segmentation,
        exclude_low_score_only_patients=config.get("exclude_low_score_only_patients", True),
    )
    if normalization_stats is not None:
        print(f"Using training-set normalization parameters: {normalization_stats}")
    elif not config.get("grouped_normalization", True):
        print("Grouped training-set normalization is disabled.")

    segmentation_labels = get_segmentation_labels(
        binary_segmentation, config.get("exclude_low_score_only_patients", True)
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
    per_patient_dice = {}
    per_patient_metrics = {}
    per_patient_binary_dice = {}
    evaluated_patients = 0
    segmentation_patients = 0
    skipped_patients = 0
    evaluated_indices = {}

    def predict(sample, return_class_probabilities=False):
        batch = collate_fn([sample])
        image = batch["image"].to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(image)["segmentation_logits"].float()
        if binary_segmentation:
            probability = torch.sigmoid(logits)[0, 0]
            prediction = (probability >= 0.5).to(torch.int64)
            probabilities = (
                torch.stack((1 - probability, probability))
                if return_class_probabilities else probability
            )
        else:
            class_probabilities = torch.softmax(logits, dim=1)[0]
            prediction = class_probabilities.argmax(dim=0)
            probabilities = (
                class_probabilities if return_class_probabilities else 1 - class_probabilities[0]
            )
        return batch, prediction.cpu().numpy(), probabilities.cpu().numpy()

    for index, record in enumerate(records):
        if not dataset.has_required_volumes(record):
            print(f"[skip] patient {record.series_id}: missing PET or MRI_T2 volume")
            skipped_patients += 1
            continue
        try:
            sample = dataset[index]
        except (FileNotFoundError, ValueError) as error:
            print(f"[skip] patient {record.series_id}: {error}")
            skipped_patients += 1
            continue

        batch, prediction, _ = predict(sample)
        evaluated_patients += 1
        evaluated_indices[str(record.series_id)] = index

        segmentation_valid = bool(batch["segmentation_valid"][0])
        if segmentation_valid:
            target = batch["segmentation"][0].numpy()
            if binary_segmentation:
                target = target[0]
            target = target.astype(np.int64, copy=False)
            encoded_pairs = (
                target.reshape(-1) * segmentation_classes + prediction.reshape(-1)
            )
            patient_confusion = np.bincount(
                encoded_pairs,
                minlength=segmentation_classes * segmentation_classes,
            ).reshape(segmentation_classes, segmentation_classes)
            segmentation_confusion += patient_confusion
            patient_metrics = segmentation_metrics(patient_confusion, segmentation_labels)
            patient = str(record.series_id)
            per_patient_metrics[patient] = patient_metrics
            per_patient_dice[patient] = patient_metrics["mean_foreground_dice"]
            per_patient_binary_dice[patient] = patient_metrics["foreground_dice"]
            segmentation_patients += 1
        else:
            print(f"[info] patient {record.series_id}: no valid lesion masks; excluded from segmentation metrics")

    if not evaluated_patients:
        raise RuntimeError("No patients with readable MRI and PET volumes were available for evaluation.")

    exported_examples = []
    if segmentations_dir is not None:
        selected = select_example_patients(
            per_patient_binary_dice, list(evaluated_indices), num_examples, example_selection
        )
        for patient in selected:
            # Re-run inference on the few exported patients instead of keeping every volume in memory.
            sample = dataset[evaluated_indices[patient]]
            _, prediction, probabilities = predict(sample, per_class_probability_maps)
            if per_class_probability_maps:
                probability_maps = {
                    f"{label}_probability.nii.gz": probabilities[index]
                    for index, label in enumerate(segmentation_labels)
                }
            else:
                probability_maps = {"foreground_probability.nii.gz": probabilities}
            target_valid = bool(sample["segmentation_valid"])
            if not target_valid:
                print(f"[warning] patient {patient}: exported target is incomplete; "
                      "see target_valid=false in examples.json")
            case_dir = segmentations_dir / f"patient_{patient}"
            if save_visualization_case(
                case_dir, data_dir / patient, prediction, probability_maps, sample["segmentation"],
            ):
                exported_examples.append({
                    "patient_id": patient,
                    "foreground_dice": per_patient_binary_dice.get(patient),
                    "per_class_dice": {
                        label: per_patient_metrics[patient][label]["dice"]
                        if patient in per_patient_metrics else None
                        for label in segmentation_labels
                    },
                    "mean_foreground_dice": per_patient_dice.get(patient),
                    "path": str(case_dir),
                    "target_valid": target_valid,
                })
        if exported_examples:
            with (segmentations_dir / "examples.json").open("w", encoding="utf-8") as stream:
                json.dump({
                    "selection": example_selection,
                    "class_labels": list(segmentation_labels),
                    "examples": exported_examples,
                }, stream, indent=2)

    patient_dice_values = [value for value in per_patient_dice.values() if value is not None]
    return {
        "evaluated_patients": evaluated_patients,
        "segmentation_patients": segmentation_patients,
        "skipped_patients": skipped_patients,
        "segmentation": {
            "class_labels": list(segmentation_labels),
            "confusion_matrix": segmentation_confusion.tolist(),
            "classes": segmentation_metrics(segmentation_confusion, segmentation_labels),
            "per_patient_foreground_dice": per_patient_dice,
            "mean_patient_foreground_dice": (
                float(np.mean(patient_dice_values)) if patient_dice_values else None
            ),
        },
        "exported_examples": exported_examples,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    binary_segmentation = config["binary_segmentation"] = config.get("binary_segmentation", False)

    experiment_dir = resolve_path(config["experiment_dir"])
    k_folds = config_k_folds(config)
    if args.fold is not None:
        if k_folds is None:
            raise ValueError("--fold requires k_folds greater than 1 in the config file.")
        if not 1 <= args.fold <= k_folds:
            raise ValueError(f"--fold must be between 1 and {k_folds}.")
        run_dir = experiment_dir / f"fold_{args.fold}"
    elif k_folds is not None and not args.checkpoint:
        raise ValueError(
            f"The config uses k_folds={k_folds}; pass --fold N to evaluate one fold, "
            "or run Training/evaluate_kfold.py to aggregate all folds."
        )
    else:
        run_dir = experiment_dir
    artifact_id = f"{config['model_name']}_{config['model_version']}"
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else (
        run_dir / f"{artifact_id}.pt"
    )
    data_dir = resolve_path(args.data_dir or config["data_dir"])
    selected_split = args.split or config.get("split", "test")
    output_dir = resolve_path(args.output_dir) if args.output_dir else run_dir / "evaluation" / config["split"]
    output_dir.mkdir(parents=True, exist_ok=True)
    segmentations_dir = output_dir / "predicted_segmentations"
    history_path = resolve_path(args.history) if args.history else (
        run_dir / f"{artifact_id}_history.pickle"
    )
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    loss_plot_path = output_dir / "loss_curves.png"
    plot_loss_curves(history_path, loss_plot_path)

    split_dir = run_dir / "splits"
    splits = load_split_records(run_dir)
    records = select_records(splits, selected_split)
    if not records:
        raise ValueError(f"No records selected for split '{selected_split}'.")
    num_examples = args.num_examples if args.num_examples is not None else int(config.get("num_examples", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalization_stats = load_model(config, checkpoint_path, device)
    results = evaluate_records(
        model, records, config, data_dir, normalization_stats, device,
        segmentations_dir=segmentations_dir, num_examples=num_examples,
        example_selection=args.example_selection,
    )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": selected_split,
        "fold": args.fold,
        "split_seed": int(config.get("split_seed", 42)),
        "split_patient_counts": {name: len(records) for name, records in splits.items()},
        **results,
    }
    segmentation_labels = metrics["segmentation"]["class_labels"]

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    plot_confusion_matrix(
        np.asarray(metrics["segmentation"]["confusion_matrix"]),
        output_dir / "segmentation_confusion_matrix.png",
        "Binary voxel segmentation" if binary_segmentation else "Multiclass voxel segmentation",
        segmentation_labels,
        "segmentation class",
    )

    print(
        f"Evaluated {metrics['evaluated_patients']} patients; segmentation evaluated on "
        f"{metrics['segmentation_patients']}; skipped {metrics['skipped_patients']}."
    )
    print(f"Mean foreground segmentation Dice: {metrics['segmentation']['classes']['mean_foreground_dice']}")
    print(f"Mean per-patient foreground Dice: {metrics['segmentation']['mean_patient_foreground_dice']}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Loaded split metadata CSVs from: {split_dir}")
    print(f"Saved loss curves: {loss_plot_path}")
    print(f"Saved {len(metrics['exported_examples'])} visualization cases: {segmentations_dir}")
    print(f"Saved confusion matrices: {output_dir}")


if __name__ == "__main__":
    main()
