from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ["image_path", "source", "quality_label", "stage1_label", "stage3_label"]
VALID_QUALITY = {"medium", "good"}
VALID_STAGE1 = {"abnormal", "normal"}
POSITIVE_STAGE3 = "referral"


@dataclass
class OcularSurfaceTable:
    dataframe: pd.DataFrame


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess_table(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    validate_columns(df)

    normalized = df.copy()
    for col in ["quality_label", "stage1_label", "stage3_label", "source"]:
        normalized[col] = normalized[col].astype(str).str.strip().str.lower()

    filtered = normalized[
        normalized["quality_label"].isin(VALID_QUALITY)
        & normalized["stage1_label"].isin(VALID_STAGE1)
    ].reset_index(drop=True)

    filtered["binary_label"] = (filtered["stage3_label"] == POSITIVE_STAGE3).astype(int)
    return filtered


class OcularSurfaceDataset(Dataset):
    def __init__(self, excel_path: str, image_root: str, transform: Optional[Callable] = None) -> None:
        df = pd.read_excel(excel_path)
        validate_columns(df.assign(binary_label=df.get("binary_label", 0)))
        if "binary_label" not in df.columns:
            raise ValueError("Expected a preprocessed table containing the 'binary_label' column.")

        self.paths: List[str] = []
        self.labels: List[int] = []
        self.transform = transform
        missing: List[str] = []

        for _, row in df.iterrows():
            raw_path = str(row["image_path"])
            label = int(row["binary_label"])
            if os.path.exists(raw_path):
                final_path = raw_path
            else:
                final_path = os.path.join(image_root, os.path.basename(raw_path))
                if not os.path.exists(final_path):
                    missing.append(os.path.basename(raw_path))
                    continue
            self.paths.append(final_path)
            self.labels.append(label)

        if missing:
            warnings.warn(f"Skipped {len(missing)} missing images. First five: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple:
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index], self.paths[index]
