"""Hàm không gian: bounding box, chiếu tọa độ, haversine, BallTree."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000

# Bounding box Q. Hải Châu (xấp xỉ).
HAI_CHAU_BBOX = dict(lat_min=16.00, lat_max=16.09, lon_min=108.18, lon_max=108.26)
HAI_CHAU_CENTER = (16.06, 108.22)

# Đà Nẵng (kinh độ ~108.2°E) thuộc UTM zone 49N (108°E–114°E), KHÔNG phải 48N.
UTM_EPSG = 32649
_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)


def buffered_bbox(bbox: dict = HAI_CHAU_BBOX, buffer_m: float = 1000) -> dict:
    """Nới bbox thêm buffer_m mét mỗi phía (quy đổi độ theo vĩ độ trung tâm)."""
    lat_c = (bbox["lat_min"] + bbox["lat_max"]) / 2
    dlat = buffer_m / 111_320
    dlon = buffer_m / (111_320 * np.cos(np.radians(lat_c)))
    return dict(lat_min=bbox["lat_min"] - dlat, lat_max=bbox["lat_max"] + dlat,
                lon_min=bbox["lon_min"] - dlon, lon_max=bbox["lon_max"] + dlon)


def in_bbox(df: pd.DataFrame, bbox: dict, lat="latitude", lon="longitude") -> pd.Series:
    return (df[lat].between(bbox["lat_min"], bbox["lat_max"])
            & df[lon].between(bbox["lon_min"], bbox["lon_max"]))


def to_utm(df: pd.DataFrame, lat="latitude", lon="longitude") -> np.ndarray:
    """Trả về mảng (n, 2) tọa độ mét [x, y] trong UTM 49N."""
    x, y = _to_utm.transform(df[lon].to_numpy(), df[lat].to_numpy())
    return np.column_stack([x, y])


def to_radians(df: pd.DataFrame, lat="latitude", lon="longitude") -> np.ndarray:
    """Mảng (n, 2) [lat, lon] radian, dùng cho metric haversine."""
    return np.radians(df[[lat, lon]].to_numpy())


def meters_to_rad(m: float) -> float:
    return m / EARTH_RADIUS_M


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def build_balltree(df: pd.DataFrame) -> BallTree:
    return BallTree(to_radians(df), metric="haversine")
