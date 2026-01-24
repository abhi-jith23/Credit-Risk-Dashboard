# src/metrics.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_proba, thresholds=(0.2, 0.35, 0.5)) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "default_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }

    fpr, tpr, thr = roc_curve(y_true, y_proba)
    out["roc_curve"] = {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": [float(x) for x in thr],
    }

    out["confusion_at_thresholds"] = {}
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        out["confusion_at_thresholds"][str(t)] = {
            "threshold": float(t),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": float(accuracy_score(y_true, y_hat)),
        }

    return out


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
