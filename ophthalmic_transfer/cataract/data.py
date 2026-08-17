from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ["image_name", "source", "label"]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


class CataractDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, transform=None, oversample: bool = False):
        validate_columns(df)
        self.image_root = image_root
        self.transform = transform

        df = df.copy().reset_index(drop=True)
        df["source"] = df["source"].astype(str).str.strip().str.lower()
        df["label"] = df["label"].astype(int)

        if oversample:
            counts = df["label"].value_counts().sort_index()
            max_count = int(counts.max())
            parts = []
            for label_value in counts.index:
                subset = df[df["label"] == label_value]
                replace_flag = len(subset) < max_count
                parts.append(subset.sample(max_count, replace=replace_flag, random_state=42))
            df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

        self.df = df
        self._missing_examples = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.image_root, str(row["image_name"]))
        if not os.path.exists(image_path):
            if len(self._missing_examples) < 5:
                self._missing_examples.append(image_path)
                warnings.warn(f"Missing image file: {image_path}")
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), image_path


def split_by_sources(df: pd.DataFrame, internal_sources, external_sources):
    validate_columns(df)
    data = df.copy()
    data["source"] = data["source"].astype(str).str.strip().str.lower()
    internal_set = {str(x).strip().lower() for x in internal_sources}
    external_set = {str(x).strip().lower() for x in external_sources}
    df_internal = data[data["source"].isin(internal_set)].reset_index(drop=True)
    df_external = data[data["source"].isin(external_set)].reset_index(drop=True)
    return df_internal, df_external
