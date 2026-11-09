import argparse
from pathlib import Path
import pickle
import torch
import os
import json
from datetime import datetime
from model import UNet
from training_pipeline import (
    TrainingPipeline,
    kfold_split_records,
    load_records,
    split_records,
    summarize_fold_histories,
    write_split_csvs,
)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def get_k_folds(hyperparams):
    """Return the configured number of folds, or None for a single train/validation split."""
    k_folds = hyperparams.get("k_folds")
    if k_folds is None:
        return None
    if isinstance(k_folds, bool) or not isinstance(k_folds, int) or k_folds < 1:
        raise ValueError("k_folds must be null or a positive integer in the config file.")
    return k_folds if k_folds > 1 else None


def fold_dir_name(fold):
    return f"fold_{fold}"


def train_run(
    hyperparams,
    run_dir,
    artifact_id,
    records,
    splits,
    patients_path,
    data_csv,
    device,
    hyperparams_path,
    fold=None,
):
    """Train one model on the given splits and store its artifacts in run_dir."""
    split_dir = run_dir / "splits"
    for directory in (run_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"{artifact_id}.pt"
    history_path = run_dir / f"{artifact_id}_history.pickle"
    fold_label = f" (fold {fold}/{hyperparams['k_folds']})" if fold is not None else ""
    print(f"Run artifacts{fold_label} will be stored in {run_dir}")

    split_csv_paths = write_split_csvs(data_csv, splits, split_dir)
    print(f"Split manifests will be stored in {split_dir}")
    print("Split patient counts:", {name: len(split) for name, split in splits.items()})

    print(f"\n[2/5] Building model{fold_label}...")
    model = UNet(**hyperparams["unet"]).to(device)

    print(f"\n[3/5] Setting up training pipeline{fold_label}...")
    pipeline = TrainingPipeline(
        model=model,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        lr_reduction_patience=hyperparams["lr_reduction_epochs"],
        checkpoint_path=str(checkpoint_path),
        data_csv=hyperparams["data_csv"],
        train_ratio=hyperparams.get("train_ratio", 0.8),
        validation_ratio=hyperparams.get("validation_ratio", 0.1),
        test_ratio=hyperparams.get("test_ratio", 0.1),
        split_seed=hyperparams.get("split_seed", 42),
        scheduler_config=hyperparams["scheduler"],
        loss_weights=hyperparams.get("loss_weights"),
        class_weight_power=hyperparams["class_weight_power"],
        mixed_precision=hyperparams["mixed_precision"],
        gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
        num_workers=hyperparams["num_workers"],
        grouped_normalization=hyperparams.get("grouped_normalization", True),
        binary_segmentation=hyperparams.get("binary_segmentation", False),
        data_augmentation=hyperparams.get("data_augmentation", True),
    )

    print(f"\n[4/5] Creating Pytorch dataloaders{fold_label}...")
    pipeline.create_dataloaders(
        nii_path=patients_path,
        data_csv=data_csv,
        records=records,
        splits=splits,
    )

    normalization_path = run_dir / f"{artifact_id}_normalization.json"
    with normalization_path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline.normalization_stats, handle, indent=2, sort_keys=True)
    print(f"Normalization parameters saved to {normalization_path}")

    print(f"\n[5/5] Training model{fold_label}...")
    history = pipeline.train()

    print(f"Training completed{fold_label}!")
    print(f"Best model saved to: {pipeline.checkpoint_path}")

    with open(history_path, "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"History saved to {history_path}")

    metadata = {
        "artifact_id": artifact_id,
        "model_name": hyperparams["model_name"],
        "model_version": hyperparams["model_version"],
        "fold": fold,
        "k_folds": hyperparams.get("k_folds"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "files": {
            "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            "history": str(history_path.relative_to(run_dir)),
            "hyperparameters": os.path.relpath(hyperparams_path, run_dir),
            "normalization": normalization_path.name,
            "split_manifests": {
                name: str(path.relative_to(run_dir))
                for name, path in split_csv_paths.items()
            },
        },
        "cli_args": hyperparams,
    }
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Metadata saved to {metadata_path}")

    # Release GPU memory before the next fold builds a fresh model.
    del pipeline, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def write_cv_history(experiment_dir, artifact_id, k_folds):
    """Collect every finished fold history into one cross-validation history and summary."""
    fold_histories = {}
    for fold in range(1, k_folds + 1):
        history_path = experiment_dir / fold_dir_name(fold) / f"{artifact_id}_history.pickle"
        if history_path.is_file():
            with history_path.open("rb") as handle:
                fold_histories[fold] = pickle.load(handle)

    cv_history = {"k_folds": k_folds, "folds": fold_histories}
    cv_history_path = experiment_dir / f"{artifact_id}_cv_history.pickle"
    with cv_history_path.open("wb") as handle:
        pickle.dump(cv_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {"k_folds": k_folds, **summarize_fold_histories(fold_histories)}
    summary_path = experiment_dir / f"{artifact_id}_cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Cross-validation history ({len(fold_histories)}/{k_folds} folds) saved to {cv_history_path}")
    print(f"Cross-validation summary saved to {summary_path}")
    for key in ("train_dice_at_best_epoch", "val_dice_at_best_epoch"):
        if key in summary["aggregate"]:
            stats = summary["aggregate"][key]
            print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the prostate cancer segmentation model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Training/run_config.json",
        help="JSON config shared with Training/evaluate_model.py.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        help="Exponent for inverse-frequency class weights; set class_weight_power to null in config to disable.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        help="With k_folds > 1, train only these 1-based folds (default: all folds).",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as stream:
        hyperparams = json.load(stream)

    def resolve_path(path):
        path = Path(path)
        return (path if path.is_absolute() else REPO_ROOT / path).resolve()

    class_weight_power = (
        args.class_weight_power
        if args.class_weight_power is not None
        else hyperparams.get("class_weight_power", 2.0)
    )
    grouped_normalization = hyperparams.get("grouped_normalization", True)
    if not isinstance(grouped_normalization, bool):
        raise ValueError("grouped_normalization must be a boolean in the config file.")
    binary_segmentation = hyperparams.get("binary_segmentation", False)
    if not isinstance(binary_segmentation, bool):
        raise ValueError("binary_segmentation must be a boolean in the config file.")
    if class_weight_power is not None and class_weight_power <= 0:
        raise ValueError("class_weight_power must be greater than zero or null.")
    hyperparams["class_weight_power"] = class_weight_power
    k_folds = get_k_folds(hyperparams)
    hyperparams["k_folds"] = k_folds
    if args.folds and k_folds is None:
        raise ValueError("--folds requires k_folds greater than 1 in the config file.")
    selected_folds = sorted(set(args.folds)) if args.folds else list(range(1, (k_folds or 0) + 1))
    invalid_folds = [fold for fold in selected_folds if not 1 <= fold <= (k_folds or 0)]
    if invalid_folds:
        raise ValueError(f"Folds must be between 1 and {k_folds}: {invalid_folds}")
    model_name = hyperparams["model_name"]
    model_version = hyperparams["model_version"]

    artifact_id = f"{model_name}_{model_version}"
    experiment_dir = resolve_path(hyperparams["experiment_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment artifacts will be stored in {experiment_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    patients_path = resolve_path(hyperparams["data_dir"])
    data_csv = resolve_path(hyperparams["data_csv"])
    split_seed = int(hyperparams.get("split_seed", 42))

    print("\n[1/5] Loading records metadata...")
    records = load_records(data_csv)
    if not records:
        raise ValueError(f"No records found in {data_csv}")
    if binary_segmentation:
        eligible_primary_scores = {3, 4, 5}
        filtered_records = [
            record
            for record in records
            if any(lesion.label in eligible_primary_scores for lesion in record.lesions)
        ]
        excluded_count = len(records) - len(filtered_records)
        if excluded_count:
            print(
                "[Info] Binary segmentation enabled: excluding "
                f"{excluded_count} patient(s) without PRIMARY score 3-5 lesions."
            )
        records = filtered_records
        if not records:
            raise ValueError(
                "Binary segmentation requires patients with PRIMARY scores 3, 4, or 5. "
                f"None were found in {data_csv}."
            )

    # The held-out test set is identical in single-split and k-fold mode for the same seed.
    splits = split_records(
        records,
        float(hyperparams.get("train_ratio", 0.7)),
        float(hyperparams.get("validation_ratio", 0.15)),
        float(hyperparams.get("test_ratio", 0.15)),
        split_seed,
    )

    unet_kwargs = dict(hyperparams.get("unet", {}))
    if binary_segmentation:
        unet_kwargs["out_channels"] = 1
    hyperparams["unet"] = unet_kwargs

    hyperparams_path = experiment_dir / f"{model_name}_hyperparameters.json"
    with open(hyperparams_path, "w", encoding="utf-8") as handle:
        json.dump(hyperparams, handle, indent=2, sort_keys=True)
    print(f"Hyperparameters saved to {hyperparams_path}")

    if k_folds is None:
        train_run(
            hyperparams, experiment_dir, artifact_id, records, splits,
            patients_path, data_csv, device, hyperparams_path,
        )
        return

    folds = kfold_split_records(records, k_folds, split_seed, test_records=splits["test"])
    fold_manifest = {
        "k_folds": k_folds,
        "split_seed": split_seed,
        "test": sorted(record.series_id for record in splits["test"]),
        "folds": {
            str(index): {
                "train": sorted(record.series_id for record in fold["train"]),
                "validation": sorted(record.series_id for record in fold["validation"]),
            }
            for index, fold in enumerate(folds, start=1)
        },
    }
    fold_manifest_path = experiment_dir / "folds.json"
    with fold_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(fold_manifest, handle, indent=2)
    print(
        f"{k_folds}-fold cross-validation: held-out test={len(splits['test'])} patients, "
        f"validation sizes={[len(fold['validation']) for fold in folds]}; "
        f"fold manifest saved to {fold_manifest_path}"
    )

    for fold in selected_folds:
        print(f"\n===== Fold {fold}/{k_folds} =====")
        train_run(
            hyperparams, experiment_dir / fold_dir_name(fold), artifact_id, records,
            folds[fold - 1], patients_path, data_csv, device, hyperparams_path, fold=fold,
        )
        write_cv_history(experiment_dir, artifact_id, k_folds)


if __name__ == "__main__":
    main()
