"""Phân cụm không gian: K-Means (UTM), DBSCAN (haversine), HDBSCAN; đánh giá & mô tả cụm."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from src import geo

RANDOM_STATE = 42


def kmeans_scan(xy: np.ndarray, ks=range(2, 21)) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(xy)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(xy, km.labels_),
                         davies_bouldin=davies_bouldin_score(xy, km.labels_)))
    return pd.DataFrame(rows)


def k_distance(df: pd.DataFrame, k: int) -> np.ndarray:
    """Khoảng cách (m) tới láng giềng thứ k (không tính chính điểm), sắp tăng dần."""
    X = geo.to_radians(df)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine", algorithm="ball_tree").fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, k] * geo.EARTH_RADIUS_M)


def dbscan_haversine(df: pd.DataFrame, eps_m: float, min_samples: int) -> np.ndarray:
    X = geo.to_radians(df)
    return DBSCAN(eps=geo.meters_to_rad(eps_m), min_samples=min_samples,
                  metric="haversine", algorithm="ball_tree").fit_predict(X)


def hdbscan_utm(xy: np.ndarray, min_cluster_size: int, min_samples: int | None = None) -> np.ndarray:
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(xy)


def cluster_metrics(xy: np.ndarray, labels: np.ndarray) -> dict:
    """Số cụm, % nhiễu, Silhouette & Davies–Bouldin (tính trên điểm KHÔNG nhiễu, trong mét)."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = dict(n_clusters=n_clusters, noise_pct=100 * (~mask).mean(),
               silhouette=np.nan, davies_bouldin=np.nan,
               largest_cluster_pct=np.nan)
    if n_clusters >= 2:
        out["silhouette"] = silhouette_score(xy[mask], labels[mask])
        out["davies_bouldin"] = davies_bouldin_score(xy[mask], labels[mask])
    if n_clusters >= 1:
        out["largest_cluster_pct"] = 100 * Counter(labels[mask]).most_common(1)[0][1] / len(labels)
    return out


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Đánh lại số cụm theo kích thước giảm dần (0 = cụm lớn nhất), giữ -1 là nhiễu."""
    cnt = Counter(labels[labels != -1])
    mapping = {old: new for new, (old, _) in enumerate(cnt.most_common())}
    return np.array([mapping.get(l, -1) for l in labels])


def profile_clusters(df: pd.DataFrame, label_col: str, type_col="type_group",
                     street_col="street", ward_col="ward_final") -> pd.DataFrame:
    """Mô tả từng cụm: số cửa hàng, tâm, bán kính, loại hình chủ đạo, rating TB, đường/phường tiêu biểu."""
    rows = []
    for c, g in df[df[label_col] != -1].groupby(label_col):
        xy = geo.to_utm(g)
        center = xy.mean(axis=0)
        r = np.sqrt(((xy - center) ** 2).sum(axis=1))
        lat_c, lon_c = geo.from_utm(center[None, :])[0]
        tg = g[type_col].value_counts(normalize=True)
        streets = g[street_col].value_counts().head(3)
        rows.append(dict(
            cluster=c, n_stores=len(g), center_lat=lat_c, center_lon=lon_c,
            radius_p90_m=np.percentile(r, 90),
            density_per_ha=len(g) / max(np.pi * (np.percentile(r, 90) / 100) ** 2, 0.01),
            top_type=tg.index[0], top_type_pct=100 * tg.iloc[0],
            second_type=tg.index[1] if len(tg) > 1 else None,
            second_type_pct=100 * tg.iloc[1] if len(tg) > 1 else np.nan,
            pct_rated=100 * g["has_rating"].mean(), mean_rating=g["rating"].mean(),
            pct_chain=100 * g["is_chain"].mean(),
            top_streets=", ".join(f"{s} ({n})" for s, n in streets.items()),
            main_ward=g[ward_col].value_counts().index[0],
        ))
    return pd.DataFrame(rows).sort_values("n_stores", ascending=False).reset_index(drop=True)


def hull_latlon(g: pd.DataFrame) -> list | None:
    """Bao lồi của cụm (list [lat, lon]) để vẽ polygon; None nếu < 3 điểm phân biệt."""
    pts = g[["latitude", "longitude"]].drop_duplicates().to_numpy()
    if len(pts) < 3:
        return None
    try:
        h = ConvexHull(pts)
    except Exception:
        return None
    return pts[h.vertices].tolist()
