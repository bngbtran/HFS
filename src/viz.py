"""Hàm vẽ dùng chung: style matplotlib, lưu hình, bản đồ folium."""
from __future__ import annotations

from pathlib import Path

import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
MAP_DIR = ROOT / "outputs" / "maps"
TABLE_DIR = ROOT / "outputs" / "tables"

# Bảng màu phân loại (thứ tự cố định, đã kiểm định cho mù màu ở các cặp kề nhau).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRAY = "#8a8985"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
CATEGORY_COLORS = {"FOOD": PALETTE[0], "SHOP": PALETTE[1]}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",  # hỗ trợ đầy đủ dấu tiếng Việt
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelcolor": TEXT_2, "axes.edgecolor": "#c9c8c2",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "legend.frameon": False,
    })


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    return path


def barh_counts(series: pd.Series, title: str, xlabel: str, ylabel: str = "",
                ax=None, color=PALETTE[0], annotate=True):
    """Bar ngang cho value_counts (đã sắp xếp), có nhãn số ở đầu thanh."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(series))))
    s = series.iloc[::-1]
    bars = ax.barh(s.index.astype(str), s.values, color=color, height=0.7)
    if annotate:
        ax.bar_label(bars, padding=3, fontsize=8, color=TEXT_2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", visible=False)
    return ax


def base_map(center=(16.06, 108.22), zoom=14, tiles="OpenStreetMap") -> folium.Map:
    return folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=True)


def add_points(m: folium.Map, df: pd.DataFrame, color: str, name: str,
               popup_cols=("storeName", "address"), radius=3):
    fg = folium.FeatureGroup(name=name)
    for _, r in df.iterrows():
        popup = "<br>".join(f"<b>{c}</b>: {r[c]}" for c in popup_cols if c in r)
        folium.CircleMarker([r["latitude"], r["longitude"]], radius=radius,
                            color=color, fill=True, fill_opacity=0.7, weight=1,
                            popup=folium.Popup(popup, max_width=300)).add_to(fg)
    fg.add_to(m)
    return fg


def save_map(m: folium.Map, name: str):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    path = MAP_DIR / f"{name}.html"
    m.save(str(path))
    return path


def show_map(path, height=500):
    """Hiển thị bản đồ đã lưu bằng IFrame (không nhúng HTML vào notebook cho nhẹ)."""
    from IPython.display import IFrame
    rel = Path("..") / Path(path).relative_to(ROOT)
    return IFrame(src=rel.as_posix(), width="100%", height=height)


SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + SEQ_BLUE)


def heatmap(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, fmt: str = ".0f",
            ax=None, cbar_label: str = "", cmap=None, annotate=True, vmin=None, vmax=None):
    """Heatmap bảng (ma trận) với chú thích giá trị từng ô; màu tuần tự 1 hue."""
    if ax is None:
        _, ax = plt.subplots(figsize=(1 + 0.75 * df.shape[1], 1 + 0.42 * df.shape[0]))
    cmap = cmap or SEQ_CMAP
    im = ax.imshow(df.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=40, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.grid(False)
    if annotate:
        vals = df.to_numpy(dtype=float)
        hi = np.nanmax(vals) if vmax is None else vmax
        lo = np.nanmin(vals) if vmin is None else vmin
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                dark = (v - lo) / (hi - lo + 1e-9) > 0.6
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7.5,
                        color="white" if dark else TEXT)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def add_heat(m: folium.Map, df: pd.DataFrame, name="Mật độ (HeatMap)", radius=14):
    from folium.plugins import HeatMap
    HeatMap(df[["latitude", "longitude"]].to_numpy().tolist(), name=name, radius=radius,
            blur=12, min_opacity=0.3).add_to(m)
