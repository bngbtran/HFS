"""Tạo đặc trưng: one-hot đa nhãn, nhóm loại hình chính, đặc trưng không gian (BallTree haversine)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import geo

# ---------------------------------------------------------------- one-hot đa nhãn
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


# ---------------------------------------------------------------- nhóm loại hình chính
SHOP_GROUP = "Cửa hàng thực phẩm"
OTHER_GROUP = "Khác"
LABEL_TO_GROUP = {
    "Coffee - Tea - Juice": "Cà phê & nước",
    "Milk Tea": "Trà sữa",
    "Noodles & Congee": "Bún/Phở/Cháo",
    "Rice": "Cơm",
    "Snack": "Ăn vặt & Bánh mì",
    "Bread": "Ăn vặt & Bánh mì",
    "Hotpot & Grill": "Lẩu & Nướng",
    "Dessert": "Tráng miệng",
    "International Food": "Fast food & Quốc tế",
    "Fast Food": "Fast food & Quốc tế",
    "Pizza": "Fast food & Quốc tế",
    "Chicken": "Fast food & Quốc tế",
    "Beef": "Fast food & Quốc tế",
    "Seafood": "Fast food & Quốc tế",
    "Japanese": "Fast food & Quốc tế",
    "Korean": "Fast food & Quốc tế",
    "Healthy Food": OTHER_GROUP,
    "Others": OTHER_GROUP,
    "Grocery": SHOP_GROUP,
    "Other Store": SHOP_GROUP,
}
GROUP_ORDER = ["Cà phê & nước", "Bún/Phở/Cháo", "Ăn vặt & Bánh mì", "Cơm", "Lẩu & Nướng",
               "Trà sữa", "Fast food & Quốc tế", "Tráng miệng", SHOP_GROUP]
GROUP_SLUG = {"Cà phê & nước": "cafe", "Bún/Phở/Cháo": "noodle", "Ăn vặt & Bánh mì": "snack",
              "Cơm": "rice", "Lẩu & Nướng": "hotpot", "Trà sữa": "milktea",
              "Fast food & Quốc tế": "intl", "Tráng miệng": "dessert", SHOP_GROUP: "shop",
              OTHER_GROUP: "other"}


def primary_group(types: str | list, category: str) -> str:
    """Nhóm loại hình chính = nhóm của nhãn đầu tiên không phải 'Others'/'Healthy Food'.

    Cửa hàng SHOP luôn thuộc 'Cửa hàng thực phẩm'. Không có nhãn cụ thể -> 'Khác'.
    """
    if category == "SHOP":
        return SHOP_GROUP
    labels = types.split("|") if isinstance(types, str) else list(types)
    for l in labels:
        g = LABEL_TO_GROUP.get(l, OTHER_GROUP)
        if g not in (OTHER_GROUP, SHOP_GROUP):
            return g
    return OTHER_GROUP


# ---------------------------------------------------------------- đặc trưng không gian
def neighbor_indices(tree, X_rad: np.ndarray, radius_m: float):
    return tree.query_radius(X_rad, r=geo.meters_to_rad(radius_m))


def spatial_features(stores: pd.DataFrame, query: pd.DataFrame | None = None,
                     radii=(100, 300, 500), group_radius=300, group_col="type_group",
                     hotspot_centers: np.ndarray | None = None) -> pd.DataFrame:
    """Đặc trưng ngữ cảnh không gian cho từng điểm truy vấn.

    stores : các cửa hàng hiện có (môi trường).
    query  : các điểm cần tính đặc trưng. None -> chính các cửa hàng, và khi đó
             LOẠI CHÍNH ĐIỂM ĐÓ khỏi mọi phép đếm (không loại các quán khác trùng tọa độ).
    Không dùng thông tin loại hình của chính điểm truy vấn -> không rò rỉ nhãn.
    """
    self_mode = query is None
    query = stores if self_mode else query
    tree = geo.build_balltree(stores)
    Xq = geo.to_radians(query)
    groups = stores[group_col].to_numpy()
    chain = stores["is_chain"].to_numpy().astype(float)
    rating = stores["rating"].to_numpy(dtype=float)
    out = pd.DataFrame(index=query.index)

    def _drop_self(ind_list):
        if not self_mode:
            return ind_list
        return [ind[ind != i] for i, ind in enumerate(ind_list)]

    for r in radii:
        ind = _drop_self(neighbor_indices(tree, Xq, r))
        out[f"n_total_{r}m"] = [len(i) for i in ind]
        if r == group_radius:
            for g in GROUP_ORDER + [OTHER_GROUP]:
                cnt = np.array([(groups[i] == g).sum() for i in ind])
                out[f"n_{GROUP_SLUG[g]}_{r}m"] = cnt
            tot = out[f"n_total_{r}m"].replace(0, np.nan)
            for g in GROUP_ORDER + [OTHER_GROUP]:
                out[f"share_{GROUP_SLUG[g]}_{r}m"] = (out[f"n_{GROUP_SLUG[g]}_{r}m"] / tot).fillna(0)
            out[f"chain_share_{r}m"] = [chain[i].mean() if len(i) else 0 for i in ind]
            out[f"mean_rating_{r}m"] = [np.nanmean(rating[i]) if np.isfinite(rating[i]).any() else np.nan
                                        for i in ind]
            out[f"n_types_unique_{r}m"] = [len(set(groups[i])) for i in ind]

    k = 2 if self_mode else 1
    dist, _ = tree.query(Xq, k=k)
    out["dist_nearest_m"] = dist[:, k - 1] * geo.EARTH_RADIUS_M
    out["dist_river_m"] = geo.dist_to_river_m(query["latitude"].to_numpy(), query["longitude"].to_numpy())
    if hotspot_centers is not None and len(hotspot_centers):
        from sklearn.neighbors import BallTree
        ht = BallTree(np.radians(hotspot_centers), metric="haversine")
        d, _ = ht.query(Xq, k=1)
        out["dist_hotspot_m"] = d[:, 0] * geo.EARTH_RADIUS_M
    out["mean_rating_300m"] = out.get("mean_rating_300m", pd.Series(np.nan, index=out.index)).fillna(
        np.nanmean(rating))
    return out


def same_type_competitors(stores: pd.DataFrame, radius_m=300, group_col="type_group") -> pd.Series:
    """Số đối thủ cùng nhóm loại hình trong bán kính (loại chính điểm đó).

    CHỈ dùng cho mô tả / chấm điểm vị trí, KHÔNG dùng làm đặc trưng khi dự đoán loại hình
    (vì phụ thuộc nhãn của chính điểm -> rò rỉ).
    """
    tree = geo.build_balltree(stores)
    ind = neighbor_indices(tree, geo.to_radians(stores), radius_m)
    g = stores[group_col].to_numpy()
    return pd.Series([(g[i[i != k]] == g[k]).sum() for k, i in enumerate(ind)], index=stores.index)
