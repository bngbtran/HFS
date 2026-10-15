"""Công cụ gợi ý (Deployment).

1) Cho một vị trí (lat, lon) -> xếp hạng 9 loại hình theo mức phù hợp.
2) Cho một loại hình -> top ô lưới nên mở quán.

Chạy (sau khi đã chạy notebook 01 → 05):
    .venv\\Scripts\\python -m src.recommend --lat 16.0678 --lon 108.2208
    .venv\\Scripts\\python -m src.recommend --type "Trà sữa" --top 5
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import cleaning as cl, features as ft, geo

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

# Trọng số điểm vị trí (tổng = 1). Giải thích trong notebook 06.
WEIGHTS = dict(demand=0.35, low_competition=0.30, colocation=0.15, model_fit=0.20)
COMPETITOR_RADIUS_M = 300
MIN_STORES_300M = 10   # ô phải có ít nhất ngần này cửa hàng trong 300 m (có "chợ" F&B)


@lru_cache(maxsize=1)
def load_assets():
    stores = pd.read_csv(PROC / "stores_clean.csv")
    stores["type_group"] = [ft.primary_group(t, c) for t, c in zip(stores["product_types"], stores["category"])]
    clusters = pd.read_csv(PROC / "stores_clusters.csv")
    stores = stores.merge(clusters, on="storeId", how="left")
    bundle = joblib.load(ROOT / "outputs" / "models" / "type_model.joblib")
    centers = pd.read_csv(TABLES / "cluster_profiles.csv")[["center_lat", "center_lon"]].to_numpy()
    lift = pd.read_csv(TABLES / "04_colocation_lift_matrix_R100.csv", index_col=0)
    return stores, bundle, centers, lift


def context_features(points: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng ngữ cảnh cho các điểm bất kỳ (cùng định nghĩa với notebook 05)."""
    stores, bundle, centers, _ = load_assets()
    f = ft.spatial_features(stores, query=points, hotspot_centers=centers)
    # điểm mới coi là trong hotspot nếu thỏa điều kiện lõi DBSCAN: ≥ 25 cửa hàng trong 100 m
    f["is_hotspot"] = (f["n_total_100m"] >= 25).astype(int)
    tree = geo.build_balltree(stores)
    _, idx = tree.query(geo.to_radians(points), k=1)
    ward = stores["ward_final"].to_numpy()[idx[:, 0]]     # phường của cửa hàng gần nhất
    for c in bundle["ward_columns"]:
        f[c] = 0
    for i, w in enumerate(ward):
        col = "ward_" + cl.to_ascii(w).replace(" ", "_")
        if col in f.columns:
            f.iloc[i, f.columns.get_loc(col)] = 1
    f["ward_guess"] = ward
    return f


def colocation_support(f: pd.DataFrame, lift: pd.DataFrame) -> pd.DataFrame:
    """Hỗ trợ đồng vị trí cho mỗi loại hình B: trung bình lift(near:A → self:B) trên các A có trong 300 m,
    trọng số theo số cửa hàng A (đặc trưng n_<A>_300m). Giá trị > 1 = môi trường 'hút' loại B."""
    groups = list(lift.columns)
    W = np.column_stack([f[f"n_{ft.GROUP_SLUG[a]}_300m"].to_numpy() for a in groups]).astype(float)
    tot = W.sum(axis=1, keepdims=True)
    W = np.divide(W, tot, out=np.zeros_like(W), where=tot > 0)
    S = W @ lift.loc[groups, groups].to_numpy()
    return pd.DataFrame(S, columns=groups, index=f.index)


SNAP_RADIUS_M = 100


@lru_cache(maxsize=1)
def store_context() -> pd.DataFrame:
    """Đặc trưng (đã loại chính điểm) + xác suất OOF GroupKFold của từng cửa hàng hiện có.

    Vị trí cửa hàng hiện có = các 'mặt tiền' thực tế. Dùng chúng làm điểm đánh giá để tránh lệch
    phân phối: tâm ô lưới thường rơi giữa khối nhà, có ít quán lân cận hơn hẳn dữ liệu huấn luyện.
    """
    stores, _, _, _ = load_assets()
    feats = pd.read_csv(PROC / "stores_features.csv").drop(
        columns=["storeName", "latitude", "longitude", "type_group"])
    preds = pd.read_csv(PROC / "stores_predictions.csv")
    return (stores[["storeId", "latitude", "longitude", "ward_final", "type_group"]]
            .merge(feats, on="storeId").merge(preds, on="storeId"))


def recommend_types(lat: float, lon: float) -> pd.DataFrame:
    """Xếp hạng loại hình cho 1 vị trí.

    Có cửa hàng trong SNAP_RADIUS_M -> dùng bối cảnh + xác suất OOF của vị trí mặt tiền gần nhất.
    Không có -> tính đặc trưng ngay tại điểm và cảnh báo ngoại suy.
    """
    _, bundle, _, lift = load_assets()
    classes = list(bundle["classes"])
    sc = store_context()
    pt = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    dist, idx = geo.build_balltree(sc).query(geo.to_radians(pt), k=1)
    d_m = dist[0, 0] * geo.EARTH_RADIUS_M
    if d_m <= SNAP_RADIUS_M:
        f = sc.iloc[[idx[0, 0]]].reset_index(drop=True)
        proba = f[[f"proba_{ft.GROUP_SLUG[c]}" for c in classes]].to_numpy()[0]
        mode = f"gắn vào vị trí cửa hàng gần nhất (cách {d_m:.0f} m)"
        ward = f["ward_final"].iloc[0]
    else:
        f = context_features(pt)
        proba = bundle["model"].predict_proba(f[bundle["features"]])[0]
        mode = f"NGOẠI SUY – không có cửa hàng nào trong {SNAP_RADIUS_M} m (gần nhất {d_m:.0f} m)"
        ward = f["ward_guess"].iloc[0]
    co = colocation_support(f, lift).iloc[0]
    out = pd.DataFrame({
        "loai_hinh": classes,
        "xac_suat_mo_hinh": proba,
        "phu_hop_lift": proba / bundle["prior"],
        "doi_thu_cung_loai_300m": [int(f[f"n_{ft.GROUP_SLUG[c]}_300m"].iloc[0]) for c in classes],
        "ho_tro_dong_vi_tri": [co[c] for c in classes],
    }).sort_values("phu_hop_lift", ascending=False).reset_index(drop=True)
    out.attrs["context"] = dict(mode=mode, n_total_300m=int(f["n_total_300m"].iloc[0]),
                                n_total_100m=int(f["n_total_100m"].iloc[0]), ward=ward)
    return out


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def score_grid(cell_m: float = 200, min_stores: int = 2) -> pd.DataFrame:
    """Chấm điểm ô lưới cho mọi loại hình. Trả về bảng dài (ô × loại hình).

    Mỗi ô được đánh giá bằng TRUNG BÌNH bối cảnh của các vị trí cửa hàng có trong ô
    (đặc trưng đã loại chính điểm; xác suất mô hình là OOF GroupKFold – không rò rỉ).
    """
    _, bundle, _, lift = load_assets()
    classes = list(bundle["classes"])
    sc = store_context().copy()
    sc["cell"] = geo.grid_id(sc, cell_m).to_numpy()
    num = [c for c in sc.columns if c.startswith(("n_", "share_", "proba_"))]
    agg = sc.groupby("cell").agg(n_stores_in_cell=("storeId", "size"),
                                 ward_guess=("ward_final", lambda s: s.mode().iloc[0]),
                                 **{c: (c, "mean") for c in num})
    agg = agg[(agg["n_stores_in_cell"] >= min_stores) & (agg["n_total_300m"] >= MIN_STORES_300M)].reset_index()
    bounds = [geo.cell_bounds(c, cell_m) for c in agg["cell"]]
    agg["lat_min"], agg["lon_min"], agg["lat_max"], agg["lon_max"] = zip(*[b[0] for b in bounds])
    agg["latitude"], agg["longitude"] = zip(*[b[1] for b in bounds])
    co = colocation_support(agg, lift)
    long = []
    for k, c in enumerate(classes):
        slug = ft.GROUP_SLUG[c]
        d = agg[["cell", "latitude", "longitude", "lat_min", "lon_min", "lat_max", "lon_max", "ward_guess",
                 "n_stores_in_cell", "n_total_100m", "n_total_300m"]].copy()
        d["loai_hinh"] = c
        d["doi_thu_cung_loai_300m"] = agg[f"n_{slug}_300m"].round(1)
        d["ty_le_cung_loai_300m"] = agg[f"share_{slug}_300m"]
        d["ho_tro_dong_vi_tri"] = co[c].to_numpy()
        d["phu_hop_lift"] = agg[f"proba_{slug}"] / bundle["prior"][k]
        # các thành phần điểm, chuẩn hóa bằng phân vị trong cùng loại hình -> [0, 1]
        d["s_demand"] = _pct_rank(d["n_total_300m"])
        d["s_low_competition"] = 1 - _pct_rank(d["ty_le_cung_loai_300m"])
        d["s_colocation"] = _pct_rank(d["ho_tro_dong_vi_tri"])
        d["s_model_fit"] = _pct_rank(d["phu_hop_lift"])
        d["score"] = (WEIGHTS["demand"] * d["s_demand"] + WEIGHTS["low_competition"] * d["s_low_competition"]
                      + WEIGHTS["colocation"] * d["s_colocation"] + WEIGHTS["model_fit"] * d["s_model_fit"])
        long.append(d)
    out = pd.concat(long, ignore_index=True)
    out["rank_in_type"] = out.groupby("loai_hinh")["score"].rank(ascending=False, method="first").astype(int)
    return out


def recommend_locations(type_group: str, top: int = 5, cell_m: float = 200) -> pd.DataFrame:
    g = score_grid(cell_m)
    if type_group not in set(g["loai_hinh"]):
        raise ValueError(f"Loại hình không hợp lệ. Chọn một trong: {sorted(set(g['loai_hinh']))}")
    return g[g["loai_hinh"] == type_group].sort_values("score", ascending=False).head(top).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gợi ý loại hình / vị trí F&B ở Q. Hải Châu")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--type", dest="type_group", help=f"Một trong: {', '.join(ft.GROUP_ORDER)}")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.lat is not None and a.lon is not None:
        r = recommend_types(a.lat, a.lon)
        print(f"Vị trí ({a.lat}, {a.lon}) – bối cảnh: {r.attrs['context']}")
        print(r.round(3).to_string(index=False))
    elif a.type_group:
        r = recommend_locations(a.type_group, a.top)
        cols = ["rank_in_type", "cell", "latitude", "longitude", "ward_guess", "n_stores_in_cell", "n_total_300m",
                "doi_thu_cung_loai_300m", "ho_tro_dong_vi_tri", "phu_hop_lift", "score"]
        print(f"Top {a.top} ô 200 m cho loại hình '{a.type_group}':")
        print(r[cols].round(3).to_string(index=False))
    else:
        ap.print_help()
