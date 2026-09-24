from pathlib import Path
import pickle
import torch
import os
import torch.distributed as dist
import json
from datetime import datetime
import numpy as np
from model import UNet
from training_pipeline import TrainingPipeline, load_patch_records
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODE_ENV_VAR = "TRAINING_MODE"

HYPERPARAMS = {
    "data_dir": "Data/resampled_images",
    "data_csv": "Data/data.csv",
    "input_shape": (128, 128, 128),
    "batch_size": 5,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "scheduler":{
        "name" : "cosine",
        "eta_min": 1e-6
    },

    # "scheduler": {
    #     "name": "plateau",
    #     "mode": "min",
    #     "factor": 0.5,
    #     "patience": 4,
    #     "min_lr": 1e-6,
    # },

    # "scheduler": {
    #     "name": "onecycle",
    # },
    
    "train_ratio": 0.7,
    "lr_reduction_epochs": 10,
    "unet": {
        "in_channels": 1,
        "out_channels": 1,
        "n_blocks": 4,
        "start_filters": 16,
        "activation": "relu",
        "normalization": "group4",
        "conv_mode": "same",
        "up_mode": "trilinear",
        "middle_neurons": 256,
        "class_output": 1,
        "dropout": 0.2,
        "attention": True,
    },
}

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

def main():
    hyperparams = HYPERPARAMS.copy()
    model_name = os.environ.get("RUN_MODEL_NAME")
    model_version = os.environ.get("RUN_MODEL_VERSION")
    input_shape = tuple(hyperparams["input_shape"])

    # SageMaker model / output directories (or "." locally)
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "."))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "."))

    
    artifact_id = f"{model_name}_{model_version}"
    local_artifact_root = REPO_ROOT / f"{model_name}_{model_version}"
    model_dir = local_artifact_root
    output_dir = local_artifact_root
    print(f"Local artifacts will be stored in {local_artifact_root}")

    # artifact directories
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = f"{artifact_id}.pt" if artifact_id else "prostate_cancer_classification.pt"
    checkpoint_path = model_dir / checkpoint_name
    history_path = output_dir / "history.pickle"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    patients_path = Path(hyperparams["data_dir"])
    data_csv = Path(hyperparams["data_csv"])
    proportion = float(hyperparams.get("proportion_to_use", 1.0))
    split_seed = int(hyperparams.get("split_seed", 42))

    print("\n[1/5] Loading patch metadata...")
    records = load_patch_records(data_csv)
    print("initial records:")
    print(records)
    if not records:
        raise ValueError(f"No records found in {data_csv}")

    # patient-level subsampling by proportion_to_use
    series_ids = sorted({r.series_id for r in records})
    if not (0 < proportion <= 1):
        raise ValueError("proportion_to_use must be in (0,1].")
    if proportion < 1.0:
        rng = np.random.default_rng(split_seed)
        keep_count = max(1, int(round(len(series_ids) * proportion)))
        keep_series = set(rng.choice(series_ids, size=keep_count, replace=False))
        records = [r for r in records if r.series_id in keep_series]
        print(f"[Info] Using {keep_count}/{len(series_ids)} series (~{proportion*100:.1f}%).")
    else:
        keep_series = set(series_ids)

    # report counts
    # pos_count = sum(1 for r in records if r.label == 1)
    # neg_count = sum(1 for r in records if r.label == 0)
    # print(f"[Info] Patch counts -> positives: {pos_count}, negatives: {neg_count}, total: {len(records)}")
    # print(f"[Info] Series covered: {len({r.series_id for r in records})}")

    print("\n[2/5] Building model...")

    unet_kwargs = dict(hyperparams.get("unet", {}))
    model = UNet(**unet_kwargs)

    if torch.cuda.is_available():
    #     # Run the summary on CPU to avoid exhausting limited GPU VRAM
    #     print(summary(model.to("cpu"), input_size=(1,) + input_shape, device="cpu"))
        model = model.to(device)
    # else:
    #     print(summary(model, input_size=(1,) + input_shape, device="cpu"))

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
        split_seed=hyperparams.get("split_seed", 42),
        scheduler_config=hyperparams["scheduler"],
        loss_weights=hyperparams.get("loss_weights"),
    )

    print("\n[4/5] Creating Pytorch dataloaders...")
    train_dataset, val_dataset = pipeline.create_dataloaders(
        nii_path=patients_path,
        data_csv="data_csv",
        train_ratio=hyperparams.get("train_ratio", 0.8),
        split_seed=hyperparams.get("split_seed", 42),
        records=records,
    )

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
    hyperparams_path = output_dir / "hyperparameters.json"
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
                "checkpoint": checkpoint_path.name,
                "history": history_path.name,
                "hyperparameters": hyperparams_path.name,
            },
            "cli_args": hyperparams,
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()