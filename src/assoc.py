"""Luật kết hợp: tạo transaction (trong cửa hàng, lưới, lân cận) và khai phá bằng Apriori / FP-Growth."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from src import geo


def encode(transactions: list) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


def time_algorithms(oh: pd.DataFrame, supports=(0.005, 0.01, 0.02, 0.05), repeats: int = 5) -> pd.DataFrame:
    """Thời gian chạy trung bình (ms) của Apriori và FP-Growth ở các mức min_support."""
    rows = []
    for s in supports:
        for name, fn in [("Apriori", apriori), ("FP-Growth", fpgrowth)]:
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fi = fn(oh, min_support=s, use_colnames=True)
                ts.append((time.perf_counter() - t0) * 1000)
            rows.append(dict(min_support=s, algorithm=name, time_ms=np.median(ts), n_itemsets=len(fi)))
    return pd.DataFrame(rows)


def mine_rules(oh: pd.DataFrame, min_support: float, min_confidence: float,
               min_lift: float = 1.0, max_len: int | None = None) -> pd.DataFrame:
    """FP-Growth -> luật; lọc lift > min_lift; thêm cột chuỗi dễ đọc; sắp theo lift, conviction."""
    fi = fpgrowth(oh, min_support=min_support, use_colnames=True, max_len=max_len)
    if fi.empty:
        return pd.DataFrame()
    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence,
                              num_itemsets=len(oh))
    rules = rules[rules["lift"] > min_lift].copy()
    rules["antecedents_str"] = rules["antecedents"].map(lambda s: " + ".join(sorted(s)))
    rules["consequents_str"] = rules["consequents"].map(lambda s: " + ".join(sorted(s)))
    rules["n_antecedents"] = rules["antecedents"].map(len)
    rules["support_count"] = (rules["support"] * len(oh)).round().astype(int)
    rules["conviction"] = rules["conviction"].replace(np.inf, np.nan)
    keep = ["antecedents_str", "consequents_str", "n_antecedents", "antecedent support", "consequent support",
            "support", "support_count", "confidence", "lift", "leverage", "conviction", "antecedents", "consequents"]
    return rules[keep].sort_values(["lift", "conviction"], ascending=False).reset_index(drop=True)


def grid_transactions(df: pd.DataFrame, size_m: float, col: str = "type_group",
                      min_stores: int = 2) -> tuple[list, pd.DataFrame]:
    """Transaction = tập loại hình có trong 1 ô lưới size_m × size_m (chỉ giữ ô có ≥ min_stores cửa hàng)."""
    cell = geo.grid_id(df, size_m)
    g = df.groupby(cell)
    info = pd.DataFrame({"n_stores": g.size(), "items": g[col].agg(lambda s: sorted(set(s)))})
    info = info[info["n_stores"] >= min_stores]
    return info["items"].tolist(), info


def neighborhood_transactions(df: pd.DataFrame, radius_m: float, col: str = "type_group") -> list:
    """Mô hình lấy cửa hàng làm tâm (reference-feature centric, Koperski & Han 1995).

    Mỗi cửa hàng -> 1 transaction gồm 'self:<loại hình>' và 'near:<loại hình>' cho mọi loại hình
    xuất hiện trong bán kính radius_m (không tính chính cửa hàng đó).
    """
    tree = geo.build_balltree(df)
    ind = tree.query_radius(geo.to_radians(df), r=geo.meters_to_rad(radius_m))
    vals = df[col].to_numpy()
    out = []
    for k, i in enumerate(ind):
        near = {f"near:{v}" for v in vals[i[i != k]]}
        out.append(sorted({f"self:{vals[k]}"} | near))
    return out
