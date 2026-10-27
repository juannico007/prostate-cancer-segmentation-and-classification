import argparse
from pathlib import Path
import pickle
import torch
import os
import torch.distributed as dist
import json
from datetime import datetime
import numpy as np
from model import UNet
from training_pipeline import TrainingPipeline, load_records, split_records, write_split_csvs
import pandas as pd


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

def main():
    parser = argparse.ArgumentParser(description="Train the prostate cancer segmentation and classification model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Training/run_config.json",
        help="JSON config shared with Training/evaluate_model.py.",
    )
    parser.add_argument(
        "--classification-loss",
        choices=("cross_entropy", "coral", "gaussian"),
        help="Classification objective; defaults to the value in the config file.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        help="Standard deviation for Gaussian soft targets.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        help="Exponent for inverse-frequency class weights; set class_weight_power to null in config to disable.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as stream:
        hyperparams = json.load(stream)

    def resolve_path(path):
        path = Path(path)
        return (path if path.is_absolute() else REPO_ROOT / path).resolve()

    classification_loss = args.classification_loss or hyperparams.get(
        "classification_loss", "cross_entropy"
    )
    gaussian_sigma = (
        args.gaussian_sigma
        if args.gaussian_sigma is not None
        else float(hyperparams.get("gaussian_sigma", 1.0))
    )
    class_weight_power = (
        args.class_weight_power
        if args.class_weight_power is not None
        else hyperparams.get("class_weight_power", 2.0)
    )
    if class_weight_power is not None and class_weight_power <= 0:
        raise ValueError("class_weight_power must be greater than zero or null.")
    hyperparams["classification_loss"] = classification_loss
    hyperparams["gaussian_sigma"] = gaussian_sigma
    hyperparams["class_weight_power"] = class_weight_power
    model_name = hyperparams["model_name"]
    model_version = hyperparams["model_version"]

    artifact_id = f"{model_name}_{model_version}"
    experiment_dir = resolve_path(hyperparams["experiment_dir"])
    split_dir = experiment_dir / "splits"
    for directory in (experiment_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / f"{artifact_id}.pt"
    history_path = experiment_dir / f"{artifact_id}_history.pickle"
    print(f"Experiment artifacts will be stored in {experiment_dir}")

    print(f"Split manifests will be stored in {split_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    patients_path = Path(hyperparams["data_dir"])
    data_csv = Path(hyperparams["data_csv"])
    if not data_csv.is_absolute():
        data_csv = REPO_ROOT / data_csv
    patients_path = resolve_path(patients_path)
    data_csv = resolve_path(data_csv)
    split_seed = int(hyperparams.get("split_seed", 42))

    print("\n[1/5] Loading records metadata...")
    records = load_records(data_csv)
    if not records:
        raise ValueError(f"No records found in {data_csv}")

    splits = split_records(
        records,
        float(hyperparams.get("train_ratio", 0.7)),
        float(hyperparams.get("validation_ratio", 0.15)),
        float(hyperparams.get("test_ratio", 0.15)),
        split_seed,
    )
    split_csv_paths = write_split_csvs(data_csv, splits, split_dir)
    print("Split patient counts:", {name: len(split) for name, split in splits.items()})

    # report counts
    # pos_count = sum(1 for r in records if r.label == 1)
    # neg_count = sum(1 for r in records if r.label == 0)
    # print(f"[Info] Patch counts -> positives: {pos_count}, negatives: {neg_count}, total: {len(records)}")
    # print(f"[Info] Series covered: {len({r.series_id for r in records})}")

    print("\n[2/5] Building model...")

    unet_kwargs = dict(hyperparams.get("unet", {}))
    unet_kwargs["ordinal_classification"] = classification_loss == "coral"
    hyperparams["unet"] = unet_kwargs
    model = UNet(**unet_kwargs).to(device)

    print("\n[3/5] Setting up training pipeline...")
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
        classification_loss=classification_loss,
        gaussian_sigma=gaussian_sigma,
        class_weight_power=class_weight_power,
        mixed_precision=hyperparams["mixed_precision"],
        gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
        num_workers=hyperparams["num_workers"],
    )

    print("\n[4/5] Creating Pytorch dataloaders...")
    train_dataset, val_dataset = pipeline.create_dataloaders(
        nii_path=patients_path,
        data_csv=data_csv,
        train_ratio=hyperparams.get("train_ratio", 0.8),
        validation_ratio=hyperparams.get("validation_ratio", 0.1),
        test_ratio=hyperparams.get("test_ratio", 0.1),
        split_seed=hyperparams.get("split_seed", 42),
        records=records,
        splits=splits,
    )

    normalization_path = experiment_dir / f"{artifact_id}_normalization.json"
    with normalization_path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline.normalization_stats, handle, indent=2, sort_keys=True)
    print(f"Normalization parameters saved to {normalization_path}")

    print("\n[5/5] Training model...")
    history = pipeline.train()

    print("Training completed!")
    print(f"Best model saved to: {pipeline.checkpoint_path}")

    # save history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"History saved to {history_path}")

    # save hyperparameters
    hyperparams_path = experiment_dir / f"{model_name}_hyperparameters.json"
    with open(hyperparams_path, "w", encoding="utf-8") as handle:
        json.dump(hyperparams, handle, indent=2, sort_keys=True)
    print(f"Hyperparameters saved to {hyperparams_path}")

    # save metadata
    if artifact_id:
        metadata = {
            "artifact_id": artifact_id,
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "files": {
                "checkpoint": str(checkpoint_path.relative_to(experiment_dir)),
                "history": str(history_path.relative_to(experiment_dir)),
                "hyperparameters": hyperparams_path.name,
                "normalization": normalization_path.name,
                "split_manifests": {
                    name: str(path.relative_to(experiment_dir))
                    for name, path in split_csv_paths.items()
                },
            },
            "cli_args": hyperparams,
        }
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()