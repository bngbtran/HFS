"""Tạo đặc trưng: one-hot đa nhãn productType (các đặc trưng không gian bổ sung ở notebook 05)."""
from __future__ import annotations

import re

import pandas as pd


def label_to_col(label: str, prefix: str = "pt_") -> str:
    """'Coffee - Tea - Juice' -> 'pt_coffee_tea_juice'."""
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return prefix + s


def multilabel_onehot(lists: pd.Series, prefix: str = "pt_") -> pd.DataFrame:
    """One-hot đa nhãn từ cột list; cột sắp theo tần suất giảm dần."""
    exploded = lists.explode()
    order = exploded.value_counts().index
    oh = pd.crosstab(exploded.index, exploded).reindex(columns=order, fill_value=0)
    oh = (oh > 0).astype(int).reindex(lists.index, fill_value=0)
    oh.columns = [label_to_col(c, prefix) for c in oh.columns]
    oh.index.name = None
    oh.columns.name = None
    return oh


def onehot_label_map(lists: pd.Series, prefix: str = "pt_") -> dict:
    """Bảng ánh xạ tên cột one-hot -> nhãn gốc."""
    labels = lists.explode().dropna().unique()
    return {label_to_col(l, prefix): l for l in labels}
