import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_model import (
    DEFAULT_RUN_CONFIG,
    EXAMPLE_SELECTIONS,
    config_k_folds,
    evaluate_records,
    load_config,
    load_model,
    load_split_records,
    plot_confusion_matrix,
    plot_loss_curves,
    resolve_path,
    segmentation_metrics,
    select_records,
)
from training_pipeline import summarize_fold_histories


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate every k-fold cross-validation model and aggregate the metrics."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_RUN_CONFIG))
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test", "train"),
        default=["val", "test"],
        help="Splits evaluated for every fold model (default: out-of-fold validation and held-out test).",
    )
    parser.add_argument("--folds", type=int, nargs="+", help="1-based folds to evaluate (default: all).")
    parser.add_argument(
        "--num-examples",
        type=int,
        help="Patients exported for visualization per fold and split (default: num_examples in the config).",
    )
    parser.add_argument(
        "--example-selection",
        choices=EXAMPLE_SELECTIONS,
        default="spread",
        help="Which patients to export: spread over the Dice range (default), worst, best, or first.",
    )
    parser.add_argument(
        "--example-splits",
        nargs="+",
        choices=("val", "test", "train"),
        help="Splits for which examples are exported (default: all evaluated splits).",
    )
    return parser.parse_args()


def mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)), "n": len(values)}


def aggregate_split(fold_metrics):
    """Average fold-level metrics and pool voxel confusion matrices across folds."""
    labels = next(iter(fold_metrics.values()))["segmentation"]["class_labels"]
    pooled_confusion = np.sum(
        [np.asarray(metrics["segmentation"]["confusion_matrix"]) for metrics in fold_metrics.values()],
        axis=0,
    )
    per_class = {
        label: {
            metric: mean_std([
                metrics["segmentation"]["classes"][label][metric]
                for metrics in fold_metrics.values()
            ])
            for metric in ("dice", "precision", "recall")
        }
        for label in labels
    }
    all_patient_dice = [
        value
        for metrics in fold_metrics.values()
        for value in metrics["segmentation"]["per_patient_foreground_dice"].values()
    ]
    return {
        "folds": sorted(int(fold) for fold in fold_metrics),
        "mean_foreground_dice": mean_std([
            metrics["segmentation"]["classes"]["mean_foreground_dice"]
            for metrics in fold_metrics.values()
        ]),
        "mean_patient_foreground_dice": mean_std([
            metrics["segmentation"]["mean_patient_foreground_dice"]
            for metrics in fold_metrics.values()
        ]),
        "per_class": per_class,
        "pooled": {
            "class_labels": labels,
            "confusion_matrix": pooled_confusion.tolist(),
            "classes": segmentation_metrics(pooled_confusion, labels),
            "patient_foreground_dice": mean_std(all_patient_dice),
        },
    }


def plot_cv_dice_curves(fold_histories, output_path):
    """Plot per-fold train/validation Dice curves plus the across-fold mean and std."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = plt.cm.tab10.colors
    for index, (fold, history) in enumerate(sorted(fold_histories.items())):
        color = colors[index % len(colors)]
        for split, style in (("train", "--"), ("val", "-")):
            values = history.get(f"{split}_dice") or []
            axes[0].plot(
                np.arange(1, len(values) + 1), values, style, color=color,
                label=f"Fold {fold} {split}", alpha=0.85,
            )
        best_index = int(np.argmin(history["val_total"]))
        axes[0].scatter(best_index + 1, history["val_dice"][best_index], color=color, marker="o", zorder=3)
    axes[0].set_title("Dice per fold (dot = checkpointed epoch)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Soft Dice")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize="small", ncol=2)

    max_epochs = max(len(history.get("val_dice") or []) for history in fold_histories.values())
    epochs = np.arange(1, max_epochs + 1)
    for split, label in (("train", "Train"), ("val", "Validation")):
        curves = [history.get(f"{split}_dice") or [] for history in fold_histories.values()]
        means, stds = [], []
        for epoch_index in range(max_epochs):
            values = [curve[epoch_index] for curve in curves if epoch_index < len(curve)]
            means.append(np.mean(values))
            stds.append(np.std(values))
        means, stds = np.array(means), np.array(stds)
        axes[1].plot(epochs, means, label=f"{label} mean")
        axes[1].fill_between(epochs, means - stds, means + stds, alpha=0.25)
    axes[1].set_title(f"Mean +/- std Dice across {len(fold_histories)} folds")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Soft Dice")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def format_stat(stat):
    return "n/a" if stat is None else f"{stat['mean']:.4f} +/- {stat['std']:.4f}"


def main():
    args = parse_args()
    config = load_config(args.config)
    binary_segmentation = config.get("binary_segmentation", False)
    k_folds = config_k_folds(config)
    if k_folds is None:
        raise ValueError("evaluate_kfold.py requires k_folds greater than 1 in the config file.")
    folds = sorted(set(args.folds)) if args.folds else list(range(1, k_folds + 1))
    invalid_folds = [fold for fold in folds if not 1 <= fold <= k_folds]
    if invalid_folds:
        raise ValueError(f"Folds must be between 1 and {k_folds}: {invalid_folds}")

    experiment_dir = resolve_path(config["experiment_dir"])
    artifact_id = f"{config['model_name']}_{config['model_version']}"
    data_dir = resolve_path(args.data_dir or config["data_dir"])
    output_dir = resolve_path(args.output_dir) if args.output_dir else experiment_dir / "evaluation_kfold"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_examples = args.num_examples if args.num_examples is not None else int(config.get("num_examples", 4))
    example_splits = set(args.example_splits or args.splits)

    fold_histories = {}
    fold_metrics = {split: {} for split in args.splits}
    for fold in folds:
        run_dir = experiment_dir / f"fold_{fold}"
        checkpoint_path = run_dir / f"{artifact_id}.pt"
        history_path = run_dir / f"{artifact_id}_history.pickle"
        if not checkpoint_path.is_file() or not history_path.is_file():
            print(f"[skip] fold {fold}: checkpoint or history not found in {run_dir}")
            continue
        with history_path.open("rb") as stream:
            fold_histories[fold] = pickle.load(stream)
        fold_output_dir = output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        plot_loss_curves(history_path, fold_output_dir / "loss_curves.png")

        print(f"\n===== Fold {fold}/{k_folds} =====")
        splits = load_split_records(run_dir)
        model, normalization_stats = load_model(config, checkpoint_path, device)
        for split in args.splits:
            records = select_records(splits, split)
            if not records:
                print(f"[skip] fold {fold}: split '{split}' is empty")
                continue
            split_output_dir = fold_output_dir / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Evaluating fold {fold} on '{split}' ({len(records)} patients)")
            results = evaluate_records(
                model, records, config, data_dir, normalization_stats, device,
                segmentations_dir=split_output_dir / "predicted_segmentations",
                num_examples=num_examples if split in example_splits else 0,
                example_selection=args.example_selection,
            )
            metrics = {
                "checkpoint": str(checkpoint_path),
                "fold": fold,
                "split": split,
                "split_patient_counts": {name: len(split_records) for name, split_records in splits.items()},
                **results,
            }
            with (split_output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
                json.dump(metrics, stream, indent=2)
            plot_confusion_matrix(
                np.asarray(results["segmentation"]["confusion_matrix"]),
                split_output_dir / "segmentation_confusion_matrix.png",
                f"Fold {fold} {split} voxel segmentation",
                results["segmentation"]["class_labels"],
                "segmentation class",
            )
            fold_metrics[split][str(fold)] = metrics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not fold_histories:
        raise RuntimeError(f"No trained folds found in {experiment_dir}; run Training/run_training.py first.")

    summary = {
        "experiment_dir": str(experiment_dir),
        "k_folds": k_folds,
        "evaluated_folds": sorted(fold_histories),
        "binary_segmentation": binary_segmentation,
        "training_history": summarize_fold_histories(fold_histories),
        "evaluation": {},
    }
    for split, metrics_by_fold in fold_metrics.items():
        if not metrics_by_fold:
            continue
        aggregated = aggregate_split(metrics_by_fold)
        aggregated["per_fold"] = {
            fold: {
                "mean_foreground_dice": metrics["segmentation"]["classes"]["mean_foreground_dice"],
                "mean_patient_foreground_dice": metrics["segmentation"]["mean_patient_foreground_dice"],
                "segmentation_patients": metrics["segmentation_patients"],
            }
            for fold, metrics in metrics_by_fold.items()
        }
        summary["evaluation"][split] = aggregated
        pooled = aggregated["pooled"]
        plot_confusion_matrix(
            np.asarray(pooled["confusion_matrix"]),
            output_dir / f"pooled_{split}_confusion_matrix.png",
            f"Pooled {split} voxel segmentation ({len(metrics_by_fold)} folds)",
            pooled["class_labels"],
            "segmentation class",
        )

    plot_cv_dice_curves(fold_histories, output_dir / "cv_dice_curves.png")
    summary_path = output_dir / "cv_metrics.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(f"\n===== {k_folds}-fold cross-validation summary ({len(fold_histories)} folds) =====")
    history_aggregate = summary["training_history"]["aggregate"]
    print(f"Train soft Dice at checkpointed epoch: {format_stat(history_aggregate.get('train_dice_at_best_epoch'))}")
    print(f"Val soft Dice at checkpointed epoch:   {format_stat(history_aggregate.get('val_dice_at_best_epoch'))}")
    for split, aggregated in summary["evaluation"].items():
        print(
            f"[{split}] mean foreground Dice: {format_stat(aggregated['mean_foreground_dice'])} | "
            f"per-patient Dice: {format_stat(aggregated['mean_patient_foreground_dice'])} | "
            f"pooled Dice: {aggregated['pooled']['classes']['mean_foreground_dice']}"
        )
    exported = sum(
        len(metrics["exported_examples"])
        for metrics_by_fold in fold_metrics.values()
        for metrics in metrics_by_fold.values()
    )
    print(
        f"Exported {exported} visualization cases to "
        f"{output_dir}/fold_<i>/<split>/predicted_segmentations/patient_<id>/"
    )
    print(f"Saved cross-validation metrics: {summary_path}")
    print(f"Saved Dice curves: {output_dir / 'cv_dice_curves.png'}")


if __name__ == "__main__":
    main()
