from __future__ import annotations

import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ophthalmic_transfer.common.metrics import compute_binary_metrics, roc_points
from .data import CataractDataset


@torch.no_grad()
def predict_probs(model, dataloader, device):
    model.eval()
    labels_all, probs_all, paths_all = [], [], []
    for images, labels, paths in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        labels_all.extend(np.asarray(labels))
        probs_all.extend(probs)
        paths_all.extend(paths)
    return np.asarray(labels_all), np.asarray(probs_all), paths_all


def evaluate_saved_folds(fold_indices, df_internal, df_external, image_root, transform, model_paths, create_model_fn, batch_size, num_workers, device):
    all_fold_results: List[Dict] = []
    for fold_number, model_path in enumerate(model_paths, start=1):
        train_idx, val_idx = fold_indices[fold_number - 1]
        df_train = df_internal.iloc[train_idx].reset_index(drop=True)
        df_val = df_internal.iloc[val_idx].reset_index(drop=True)

        model = create_model_fn().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        result = {}
        for stage_name, stage_df in {
            "train": df_train,
            "val": df_val,
            "external": df_external,
        }.items():
            dataset = CataractDataset(stage_df, image_root=image_root, transform=transform, oversample=False)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            labels, probs, paths = predict_probs(model, loader, device)
            result[stage_name] = {"labels": labels, "probs": probs, "paths": paths}
        all_fold_results.append(result)
    return all_fold_results


def plot_roc_curves(all_fold_results, save_path: str):
    phases = ["train", "val", "external"]
    plt.figure(figsize=(15, 5))
    for idx, phase in enumerate(phases, start=1):
        plt.subplot(1, 3, idx)
        for fold_number, fold_result in enumerate(all_fold_results, start=1):
            labels = fold_result[phase]["labels"]
            probs = fold_result[phase]["probs"]
            if len(np.unique(labels)) < 2:
                continue
            fpr, tpr, auc = roc_points(labels, probs)
            plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {fold_number} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"{phase.capitalize()} ROC curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def summarize_fold_metrics(all_fold_results, save_dir: str):
    rows = []
    for fold_number, fold_result in enumerate(all_fold_results, start=1):
        for stage_name, payload in fold_result.items():
            labels = payload["labels"]
            probs = payload["probs"]
            metrics = compute_binary_metrics(labels, probs, threshold=0.5)
            rows.append({
                "fold": fold_number,
                "stage": stage_name,
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            })
            pd.DataFrame({
                "label": labels,
                "probability": probs,
                "image_path": payload["paths"],
            }).to_csv(os.path.join(save_dir, f"fold_{fold_number}_{stage_name}.csv"), index=False)

    df_metrics = pd.DataFrame(rows)
    summary = df_metrics.groupby("stage").agg(["mean", "std"])
    df_metrics.to_csv(os.path.join(save_dir, "fold_metrics.csv"), index=False)
    summary.to_csv(os.path.join(save_dir, "fold_metrics_summary.csv"))
    with open(os.path.join(save_dir, "all_fold_results.pkl"), "wb") as f:
        pickle.dump(all_fold_results, f)
    return df_metrics, summary
