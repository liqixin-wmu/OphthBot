from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from ophthalmic_transfer.common.metrics import compute_binary_metrics, roc_points, youden_threshold
from ophthalmic_transfer.common.modeling import FocalLoss, build_densenet121_binary_model, freeze_all_then_unfreeze_head
from ophthalmic_transfer.common.utils import ensure_dir, filter_trainable_parameters
from .data import OcularSurfaceDataset, preprocess_table
from .transforms import build_transforms


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    labels, probs, paths = [], [], []
    for images, y, image_paths in loader:
        images = images.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        logits = model(images)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels.extend(y.cpu().numpy())
        probs.extend(p)
        paths.extend(image_paths)
    return np.asarray(labels), np.asarray(probs), paths


@torch.no_grad()
def evaluate_model(model, loader, criterion, threshold: float, device):
    model.eval()
    loss_sum = 0.0
    labels, probs = [], []
    for images, y, _ in loader:
        images = images.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        logits = model(images)
        loss = criterion(logits, y)
        loss_sum += loss.item() * images.size(0)
        probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
    metrics = compute_binary_metrics(np.asarray(labels), np.asarray(probs), threshold=threshold)
    metrics["loss"] = loss_sum / max(len(loader.dataset), 1)
    metrics["labels"] = np.asarray(labels)
    metrics["probs"] = np.asarray(probs)
    return metrics


def _create_model(checkpoint_path: str):
    model = build_densenet121_binary_model(checkpoint_path)
    return freeze_all_then_unfreeze_head(model)


def run_pipeline(config: Dict, device: torch.device) -> None:
    input_cfg = config["input"]
    output_cfg = config["output"]
    weights_cfg = config["weights"]
    cv_cfg = config["cross_validation"]
    exp_cfg = config["experiment"]
    norm_cfg = config["normalization"]

    ensure_dir(output_cfg["split_dir"])
    ensure_dir(output_cfg["result_dir"])

    table = preprocess_table(input_cfg["excel_path"])
    filtered_path = os.path.join(output_cfg["split_dir"], "ocular_surface_filtered.csv")
    table.to_csv(filtered_path, index=False)

    external_source = str(exp_cfg["external_source"]).strip().lower()
    df_external = table[table["source"] == external_source].reset_index(drop=True)
    df_internal = table[table["source"] != external_source].reset_index(drop=True)

    external_path = os.path.join(output_cfg["split_dir"], "ocular_surface_external.csv")
    df_external.to_csv(external_path, index=False)

    df_internal["strata"] = df_internal["source"].astype(str) + "_" + df_internal["binary_label"].astype(str)
    skf = StratifiedKFold(
        n_splits=int(cv_cfg["num_folds"]),
        shuffle=True,
        random_state=int(cv_cfg["random_seed"]),
    )

    transforms = build_transforms(tuple(norm_cfg["mean"]), tuple(norm_cfg["std"]))
    ext_dataset = OcularSurfaceDataset(external_path, input_cfg["image_root"], transform=transforms["eval"])
    ext_loader = DataLoader(
        ext_dataset,
        batch_size=int(cv_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cv_cfg["num_workers"]),
        pin_memory=True,
    )

    summary_rows: List[Dict] = []
    ext_fold_rows: List[Dict] = []
    val_roc_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    probs_all_folds: List[np.ndarray] = []
    preds_all_folds: List[np.ndarray] = []
    labels_ext_global = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_internal, df_internal["strata"].values), start=1):
        print(f"\n================ Fold {fold_idx}/{cv_cfg['num_folds']} ================")
        train_df = df_internal.iloc[train_idx].reset_index(drop=True)
        val_df = df_internal.iloc[val_idx].reset_index(drop=True)

        train_path = os.path.join(output_cfg["split_dir"], f"fold_{fold_idx}_train.csv")
        val_path = os.path.join(output_cfg["split_dir"], f"fold_{fold_idx}_val.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        train_dataset = OcularSurfaceDataset(train_path, input_cfg["image_root"], transform=transforms["train"])
        val_dataset = OcularSurfaceDataset(val_path, input_cfg["image_root"], transform=transforms["eval"])

        class_counts = np.bincount(np.asarray(train_dataset.labels, dtype=int), minlength=2)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = np.asarray([class_weights[y] for y in train_dataset.labels], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=int(cv_cfg["batch_size"]), sampler=sampler, num_workers=int(cv_cfg["num_workers"]), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=int(cv_cfg["batch_size"]), shuffle=False, num_workers=int(cv_cfg["num_workers"]), pin_memory=True)

        criterion = FocalLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        pretrained_fold_dir = weights_cfg.get("pretrained_fold_model_dir")
        external_model_path = os.path.join(pretrained_fold_dir, f"best_model_fold{fold_idx}.pth") if pretrained_fold_dir else ""
        local_model_path = os.path.join(output_cfg["result_dir"], f"best_model_fold{fold_idx}.pth")

        if pretrained_fold_dir and os.path.exists(external_model_path):
            best_model_path = external_model_path
            print(f"Using existing fold model: {external_model_path}")
        else:
            model = _create_model(weights_cfg["initialization_checkpoint"]).to(device)
            optimizer = torch.optim.Adam(filter_trainable_parameters(model), lr=float(cv_cfg["learning_rate"]))
            best_auc = -1.0
            for epoch in range(1, int(cv_cfg["num_epochs"]) + 1):
                model.train()
                running_loss = 0.0
                for images, y, _ in train_loader:
                    images = images.to(device)
                    y = torch.as_tensor(y, dtype=torch.long, device=device)
                    logits = model(images)
                    loss = criterion(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)

                train_loss = running_loss / max(len(train_loader.dataset), 1)
                val_metrics = evaluate_model(model, val_loader, criterion, threshold=0.5, device=device)
                print(
                    f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_metrics['auc']:.4f} | "
                    f"val_acc@0.5={val_metrics['accuracy']:.4f} | val_sen@0.5={val_metrics['sensitivity']:.4f} | "
                    f"val_spe@0.5={val_metrics['specificity']:.4f}"
                )
                if val_metrics["auc"] > best_auc:
                    best_auc = val_metrics["auc"]
                    torch.save(model.state_dict(), local_model_path)
            best_model_path = local_model_path

        model = _create_model(weights_cfg["initialization_checkpoint"]).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        labels_val, probs_val, _ = predict_probs(model, val_loader, device)
        threshold = youden_threshold(labels_val, probs_val)
        val_roc_cache[fold_idx] = (labels_val.copy(), probs_val.copy())
        val_metrics = compute_binary_metrics(labels_val, probs_val, threshold=threshold)
        print(
            f"Validation Youden threshold={threshold:.6f} | AUC={val_metrics['auc']:.4f} | "
            f"ACC={val_metrics['accuracy']:.4f} | SEN={val_metrics['sensitivity']:.4f} | SPE={val_metrics['specificity']:.4f}"
        )

        labels_ext, probs_ext, ext_paths = predict_probs(model, ext_loader, device)
        if labels_ext_global is None:
            labels_ext_global = labels_ext.copy()
        ext_metrics = compute_binary_metrics(labels_ext, probs_ext, threshold=threshold)
        preds_ext = (probs_ext >= threshold).astype(int)

        summary_rows.append({
            "fold": fold_idx,
            "model_path": best_model_path,
            "val_auc": val_metrics["auc"],
            "val_threshold_youden": threshold,
            "val_accuracy": val_metrics["accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "external_auc": ext_metrics["auc"],
            "external_accuracy": ext_metrics["accuracy"],
            "external_sensitivity": ext_metrics["sensitivity"],
            "external_specificity": ext_metrics["specificity"],
        })
        ext_fold_rows.append({
            "fold": fold_idx,
            "threshold_from_validation": threshold,
            "auc": ext_metrics["auc"],
            "accuracy": ext_metrics["accuracy"],
            "sensitivity": ext_metrics["sensitivity"],
            "specificity": ext_metrics["specificity"],
        })
        pd.DataFrame({
            "image_path": ext_paths,
            "true_label": labels_ext,
            "probability": probs_ext,
            "prediction": preds_ext,
            "threshold_from_validation": threshold,
        }).to_csv(os.path.join(output_cfg["result_dir"], f"fold_{fold_idx}_external_predictions.csv"), index=False)

        probs_all_folds.append(probs_ext)
        preds_all_folds.append(preds_ext)

    pd.DataFrame(summary_rows).to_csv(os.path.join(output_cfg["result_dir"], "fold_results.csv"), index=False)
    pd.DataFrame(ext_fold_rows).to_csv(os.path.join(output_cfg["result_dir"], "external_metrics_per_fold.csv"), index=False)

    probs_mean = np.mean(np.stack(probs_all_folds, axis=0), axis=0)
    votes = np.sum(np.stack(preds_all_folds, axis=0), axis=0)
    majority_prediction = (votes >= (int(cv_cfg["num_folds"]) // 2 + 1)).astype(int)
    ensemble_metrics = compute_binary_metrics(labels_ext_global, probs_mean, threshold=0.5)
    ensemble_vote_metrics = compute_binary_metrics(labels_ext_global, majority_prediction.astype(float), threshold=0.5)

    pd.DataFrame([{
        "auc_probability_average": ensemble_metrics["auc"],
        "accuracy_majority_vote": ensemble_vote_metrics["accuracy"],
        "sensitivity_majority_vote": ensemble_vote_metrics["sensitivity"],
        "specificity_majority_vote": ensemble_vote_metrics["specificity"],
    }]).to_csv(os.path.join(output_cfg["result_dir"], "external_ensemble_metrics.csv"), index=False)
    pd.DataFrame({
        "image_path": ext_dataset.paths,
        "true_label": labels_ext_global,
        "probability_mean": probs_mean,
        "positive_vote_count": votes,
        "majority_vote_prediction": majority_prediction,
    }).to_csv(os.path.join(output_cfg["result_dir"], "external_ensemble_predictions.csv"), index=False)

    plt.figure(figsize=(6, 6))
    for fold_idx, (labels_val, probs_val) in val_roc_cache.items():
        fpr, tpr, auc = roc_points(labels_val, probs_val)
        plt.plot(fpr, tpr, label=f"Fold {fold_idx} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Validation ROC")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_cfg["result_dir"], "roc_validation.pdf"), format="pdf")
    plt.close()

    plt.figure(figsize=(6, 6))
    fpr, tpr, auc = roc_points(labels_ext_global, probs_mean)
    plt.plot(fpr, tpr, lw=2, label=f"External ensemble (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("External ROC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_cfg["result_dir"], "roc_external.pdf"), format="pdf")
    plt.close()
