"""Hàm vẽ dùng chung: style matplotlib, lưu hình, bản đồ folium."""
from __future__ import annotations

from pathlib import Path

import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
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
