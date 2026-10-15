"""Đánh giá mô hình phân lớp với CV thường và CV không gian (GroupKFold)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, top_k_accuracy_score)

RANDOM_STATE = 42


def oof_predict_proba(model, X: pd.DataFrame, y: np.ndarray, cv, groups=None, classes=None):
    """Xác suất out-of-fold (mỗi điểm được dự đoán bởi mô hình KHÔNG thấy nó khi huấn luyện)."""
    classes = np.unique(y) if classes is None else classes
    P = np.zeros((len(y), len(classes)))
    fold = np.full(len(y), -1)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = clone(model).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])
        # đảm bảo đúng thứ tự cột theo classes
        P[np.ix_(te, np.searchsorted(classes, m.classes_))] = p
        fold[te] = k
    return P, fold


def metrics_from_proba(y, P, classes) -> dict:
    pred = classes[P.argmax(axis=1)]
    return dict(
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        precision_macro=precision_score(y, pred, average="macro", zero_division=0),
        recall_macro=recall_score(y, pred, average="macro", zero_division=0),
        f1_macro=f1_score(y, pred, average="macro", zero_division=0),
        top3_accuracy=top_k_accuracy_score(y, P, k=3, labels=classes),
        roc_auc_ovr_macro=roc_auc_score(y, P, multi_class="ovr", average="macro", labels=classes),
    )


def compare_models(models: dict, X, y, cvs: dict, classes) -> tuple[pd.DataFrame, dict]:
    """Chạy mọi (mô hình × chiến lược CV); trả về bảng metric + dict OOF proba."""
    rows, oof = [], {}
    for cv_name, (cv, groups) in cvs.items():
        for name, model in models.items():
            t0 = time.perf_counter()
            P, _ = oof_predict_proba(model, X, y, cv, groups, classes)
            r = metrics_from_proba(y, P, classes)
            r.update(model=name, cv=cv_name, time_s=time.perf_counter() - t0)
            rows.append(r)
            oof[(name, cv_name)] = P
    cols = ["model", "cv", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
            "top3_accuracy", "roc_auc_ovr_macro", "time_s"]
    return pd.DataFrame(rows)[cols], oof


def per_class_auc(y, P, classes) -> pd.Series:
    return pd.Series({c: roc_auc_score(y == c, P[:, i]) for i, c in enumerate(classes)})
