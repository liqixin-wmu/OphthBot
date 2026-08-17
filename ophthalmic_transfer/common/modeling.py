from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        indices = torch.arange(len(targets), device=logits.device)
        pt = torch.softmax(logits, dim=1)[indices, targets].detach()
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def _clean_state_dict(state_dict):
    return OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())


def build_densenet121_binary_model(checkpoint_path: str | None = None) -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = _clean_state_dict(state_dict)
        for key in list(state_dict.keys()):
            if "classifier" in key:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)

    return model


def freeze_all_then_unfreeze_head(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def load_checkpoint_with_class_transfer(model: nn.Module, checkpoint_path: str, num_classes: int = 2, transfer_top_n: int = 1) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _clean_state_dict(state_dict)

    classifier_name = "classifier"
    in_features = model.classifier.in_features

    old_weight = None
    old_bias = None
    if transfer_top_n > 0 and f"{classifier_name}.weight" in state_dict and f"{classifier_name}.bias" in state_dict:
        old_weight = state_dict[f"{classifier_name}.weight"]
        old_bias = state_dict[f"{classifier_name}.bias"]

    for key in list(state_dict.keys()):
        if classifier_name in key:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.classifier = nn.Linear(in_features, num_classes)

    if transfer_top_n > 0 and old_weight is not None and old_bias is not None:
        n = min(transfer_top_n, old_weight.shape[0], num_classes)
        model.classifier.weight.data[:n] = old_weight[:n]
        model.classifier.bias.data[:n] = old_bias[:n]

    return model
