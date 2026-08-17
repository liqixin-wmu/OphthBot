from __future__ import annotations

import copy
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from ophthalmic_transfer.common.metrics import compute_binary_metrics
from ophthalmic_transfer.common.modeling import build_densenet121_binary_model, freeze_all_then_unfreeze_head, load_checkpoint_with_class_transfer
from ophthalmic_transfer.common.utils import ensure_dir, filter_trainable_parameters
from .data import CataractDataset, split_by_sources
from .evaluate import evaluate_saved_folds, plot_roc_curves, summarize_fold_metrics
from .transforms import build_transforms


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_probs(model, dataloader, device):
    model.eval()
    labels_all, probs_all = [], []
    for images, labels, _ in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        labels_all.extend(labels.detach().cpu().numpy())
        probs_all.extend(probs)
    return np.asarray(labels_all), np.asarray(probs_all)


def create_model_fn(checkpoint_path: str, transfer_top_n: int, freeze_backbone: bool):
    def _factory():
        model = build_densenet121_binary_model(checkpoint_path=None)
        model = load_checkpoint_with_class_transfer(model, checkpoint_path=checkpoint_path, num_classes=2, transfer_top_n=transfer_top_n)
        if freeze_backbone:
            model = freeze_all_then_unfreeze_head(model)
        return model
    return _factory


def train_one_fold(model, dataloaders, criterion, optimizer, device, num_epochs: int, save_path: str):
    start_time = time.time()
    best_metric = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    history_rows: List[Dict] = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}\n{'-' * 30}")
        epoch_row = {"epoch": epoch}
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            labels_all, probs_all = [], []
            total = 0

            for images, labels, _ in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                labels_all.extend(labels.detach().cpu().numpy())
                probs_all.extend(probs.detach().cpu().numpy())

            epoch_loss = running_loss / max(total, 1)
            metrics = compute_binary_metrics(np.asarray(labels_all), np.asarray(probs_all), threshold=0.5)
            print(
                f"{phase:<5} | Loss: {epoch_loss:.4f} | Acc: {metrics['accuracy']:.4f} | "
                f"Sens: {metrics['sensitivity']:.3f} | Spec: {metrics['specificity']:.3f} | AUC: {metrics['auc']:.3f}"
            )
            epoch_row[f"{phase}_loss"] = epoch_loss
            for key in ["accuracy", "sensitivity", "specificity", "auc"]:
                epoch_row[f"{phase}_{key}"] = metrics[key]

            if phase == "val" and metrics["accuracy"] > best_metric:
                best_metric = metrics["accuracy"]
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)

        history_rows.append(epoch_row)

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s | Best validation accuracy: {best_metric:.4f}")
    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history_rows)


def cross_validate(df_internal, image_root, create_model_fn, transforms, output_dir, stage_name, cv_cfg, device):
    labels = df_internal["label"].astype(int).values
    skf = StratifiedKFold(n_splits=int(cv_cfg["num_folds"]), shuffle=True, random_state=int(cv_cfg["random_seed"]))
    fold_indices = list(skf.split(df_internal, labels))
    rows = []

    for fold_number, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        if fold_number < int(cv_cfg.get("resume_fold", 1)):
            print(f"Skipping fold {fold_number}")
            continue

        print(f"\n================ Fold {fold_number}/{cv_cfg['num_folds']} ================")
        df_train = df_internal.iloc[train_idx].reset_index(drop=True)
        df_val = df_internal.iloc[val_idx].reset_index(drop=True)

        train_dataset = CataractDataset(df_train, image_root=image_root, transform=transforms["train"], oversample=True)
        val_dataset = CataractDataset(df_val, image_root=image_root, transform=transforms["eval"], oversample=False)
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=int(cv_cfg["batch_size"]), shuffle=True, num_workers=int(cv_cfg["num_workers"])),
            "val": DataLoader(val_dataset, batch_size=int(cv_cfg["batch_size"]), shuffle=False, num_workers=int(cv_cfg["num_workers"])),
        }

        class_weights = compute_class_weights(df_train["label"].astype(int).values, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model = create_model_fn().to(device)
        optimizer = torch.optim.Adam(filter_trainable_parameters(model), lr=float(cv_cfg["learning_rate"]))

        save_path = os.path.join(output_dir, f"fold_{fold_number}_best.pth")
        model, history = train_one_fold(model, dataloaders, criterion, optimizer, device, int(cv_cfg["num_epochs"]), save_path)
        history.to_csv(os.path.join(output_dir, f"fold_{fold_number}_history.csv"), index=False)

        val_labels, val_probs = predict_probs(model, dataloaders["val"], device)
        metrics = compute_binary_metrics(val_labels, val_probs, threshold=0.5)
        rows.append({
            "fold": fold_number,
            "model_path": save_path,
            "val_accuracy": metrics["accuracy"],
            "val_sensitivity": metrics["sensitivity"],
            "val_specificity": metrics["specificity"],
            "val_auc": metrics["auc"],
        })

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(output_dir, "cross_validation_results.csv"), index=False)
    return fold_indices, results


def run_pipeline(config: Dict, device: torch.device) -> None:
    input_cfg = config["input"]
    weights_cfg = config["weights"]
    output_cfg = config["output"]
    cv_cfg = config["cross_validation"]
    exp_cfg = config["experiment"]
    norm_cfg = config["normalization"]
    model_cfg = config["model"]

    ensure_dir(output_cfg["result_dir"])
    df = pd.read_excel(input_cfg["excel_path"])
    df_internal, df_external = split_by_sources(df, exp_cfg["internal_sources"], exp_cfg["external_sources"])
    print(f"Internal samples: {len(df_internal)}")
    print(f"External samples: {len(df_external)}")

    transforms = build_transforms(int(norm_cfg["image_size"]), tuple(norm_cfg["mean"]), tuple(norm_cfg["std"]))
    model_factory = create_model_fn(
        checkpoint_path=weights_cfg["initialization_checkpoint"],
        transfer_top_n=int(model_cfg["transfer_top_n"]),
        freeze_backbone=bool(model_cfg["freeze_backbone"]),
    )

    fold_indices, fold_results = cross_validate(
        df_internal=df_internal,
        image_root=input_cfg["image_root"],
        create_model_fn=model_factory,
        transforms=transforms,
        output_dir=output_cfg["result_dir"],
        stage_name=exp_cfg["stage_name"],
        cv_cfg=cv_cfg,
        device=device,
    )
    print(fold_results)

    model_paths = [os.path.join(output_cfg["result_dir"], f"fold_{i}_best.pth") for i in range(1, int(cv_cfg["num_folds"]) + 1)]
    all_fold_results = evaluate_saved_folds(
        fold_indices=fold_indices,
        df_internal=df_internal,
        df_external=df_external,
        image_root=input_cfg["image_root"],
        transform=transforms["eval"],
        model_paths=model_paths,
        create_model_fn=model_factory,
        batch_size=int(cv_cfg["batch_size"]),
        num_workers=int(cv_cfg["num_workers"]),
        device=device,
    )
    plot_roc_curves(all_fold_results, os.path.join(output_cfg["result_dir"], "roc_curves_train_val_external.png"))
    _, summary = summarize_fold_metrics(all_fold_results, output_cfg["result_dir"])
    print("\nMetrics summary:")
    print(summary)
