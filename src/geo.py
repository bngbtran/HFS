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


# Sông Hàn chảy gần theo hướng Bắc–Nam dọc đường Bạch Đằng; xấp xỉ bằng kinh tuyến lon ≈ 108.224.
RIVER_LON = 108.224


def dist_to_river_m(lat, lon, river_lon: float = RIVER_LON):
    """Khoảng cách (m) theo phương Đông–Tây tới đường xấp xỉ sông Hàn (lon = river_lon)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return np.abs(lon - river_lon) * 111_320 * np.cos(np.radians(lat))


_to_wgs = Transformer.from_crs(f"EPSG:{UTM_EPSG}", "EPSG:4326", always_xy=True)


def grid_index(df: pd.DataFrame, size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Chỉ số ô lưới (ix, iy) kích thước size_m trong hệ UTM 49N."""
    xy = to_utm(df)
    return np.floor(xy[:, 0] / size_m).astype(int), np.floor(xy[:, 1] / size_m).astype(int)


def grid_id(df: pd.DataFrame, size_m: float) -> pd.Series:
    ix, iy = grid_index(df, size_m)
    return pd.Series([f"{a}_{b}" for a, b in zip(ix, iy)], index=df.index)


def cell_bounds(cell: str, size_m: float):
    """(lat_min, lon_min, lat_max, lon_max) + tâm (lat, lon) của ô 'ix_iy'."""
    ix, iy = map(int, cell.split("_"))
    lon0, lat0 = _to_wgs.transform(ix * size_m, iy * size_m)
    lon1, lat1 = _to_wgs.transform((ix + 1) * size_m, (iy + 1) * size_m)
    lonc, latc = _to_wgs.transform((ix + 0.5) * size_m, (iy + 0.5) * size_m)
    return (lat0, lon0, lat1, lon1), (latc, lonc)


def from_utm(xy: np.ndarray) -> np.ndarray:
    """(n,2) UTM [x,y] -> (n,2) [lat, lon]."""
    lon, lat = _to_wgs.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lat, lon])
