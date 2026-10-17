import argparse
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from model import UNet
from run_training import HYPERPARAMS, REPO_ROOT
from training_pipeline import TrainingPipeline, load_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train segmentation first, then freeze the U-Net and train its classifier head."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--segmentation-epochs", type=int)
    parser.add_argument("--classification-epochs", type=int)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def save_history(history, path):
    with path.open("wb") as stream:
        pickle.dump(history, stream, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    config = deepcopy(HYPERPARAMS)
    data_dir = resolve_path(args.data_dir or config["data_dir"])
    data_csv = resolve_path(args.data_csv or config["data_csv"])
    output_dir = resolve_path(
        args.output_dir or "results/staged_segmentation_then_classification"
    )
    segmentation_epochs = args.segmentation_epochs or config["epochs"]
    classification_epochs = args.classification_epochs or config["epochs"]
    if segmentation_epochs < 1 or classification_epochs < 1:
        raise ValueError("Both training phases must run for at least one epoch.")

    records = load_records(data_csv)
    if not records:
        raise ValueError(f"No records found in {data_csv}")

    proportion = float(config.get("proportion_to_use", 1.0))
    if not 0 < proportion <= 1:
        raise ValueError("proportion_to_use must be in (0, 1].")
    if proportion < 1:
        series_ids = sorted({record.series_id for record in records})
        rng = np.random.default_rng(int(config.get("split_seed", 42)))
        keep_count = max(1, int(round(len(series_ids) * proportion)))
        selected_ids = set(rng.choice(series_ids, size=keep_count, replace=False))
        records = [record for record in records if record.series_id in selected_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    segmentation_checkpoint = output_dir / "segmentation_best.pt"
    classifier_checkpoint = output_dir / "classifier_best.pt"
    config["data_dir"] = str(data_dir)
    config["data_csv"] = str(data_csv)
    config["epochs"] = segmentation_epochs
    config["segmentation_epochs"] = segmentation_epochs
    config["classification_epochs"] = classification_epochs
    config["segmentation_checkpoint"] = str(segmentation_checkpoint)
    config["classifier_checkpoint"] = str(classifier_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(**config["unet"]).to(device)
    pipeline_options = {
        "batch_size": config["batch_size"],
        "weight_decay": config["weight_decay"],
        "lr_reduction_patience": config["lr_reduction_epochs"],
        "data_csv": data_csv,
        "train_ratio": config.get("train_ratio", 0.8),
        "split_seed": config.get("split_seed", 42),
        "scheduler_config": config["scheduler"],
        "mixed_precision": config["mixed_precision"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "num_workers": config["num_workers"],
    }

    #print("Phase 1: training segmentation")
    #segmentation_pipeline = TrainingPipeline(
    #    model=model,
    #    epochs=segmentation_epochs,
    #    learning_rate=config["learning_rate"],
    #    checkpoint_path=str(segmentation_checkpoint),
    #    loss_weights={"segmentation": 1.0, "classification": 0.0},
    #    training_task="segmentation",
    #    **pipeline_options,
    #)
    #segmentation_pipeline.create_dataloaders(
    #    nii_path=data_dir,
    #    data_csv=data_csv,
    #    train_ratio=config.get("train_ratio", 0.8),
    #    split_seed=config.get("split_seed", 42),
    #    records=records,
    #)
    #segmentation_history = segmentation_pipeline.train()
    #save_history(segmentation_history, output_dir / "segmentation_history.pickle")

    print("Loading best segmentation checkpoint")
    model.load_state_dict(torch.load(segmentation_checkpoint, map_location=device, weights_only=True))

    print("Phase 2: freezing the U-Net and training the classifier head")
    classifier_pipeline = TrainingPipeline(
        model=model,
        epochs=classification_epochs,
        learning_rate=config["learning_rate"],
        checkpoint_path=str(classifier_checkpoint),
        loss_weights={"segmentation": 0.0, "classification": 1.0},
        training_task="classification",
        **pipeline_options,
    )
    classifier_pipeline.create_dataloaders(
        nii_path=data_dir,
        data_csv=data_csv,
        train_ratio=config.get("train_ratio", 0.8),
        split_seed=config.get("split_seed", 42),
        records=records,
    )
    classifier_history = classifier_pipeline.train()
    save_history(classifier_history, output_dir / "classifier_history.pickle")

    with (output_dir / "hyperparameters.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
    print(f"Segmentation checkpoint: {segmentation_checkpoint}")
    print(f"Final classifier checkpoint: {classifier_checkpoint}")
    print(f"Artifacts and phase histories: {output_dir}")


if __name__ == "__main__":
    main()