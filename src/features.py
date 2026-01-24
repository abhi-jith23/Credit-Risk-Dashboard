# src/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def get_feature_names_from_preprocessor(preprocessor) -> list[str]:
    """
    Works with sklearn ColumnTransformer using get_feature_names_out.
    """
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        # Fallback: return generic names if something changes
        return []


def logistic_coefficients_table(pipeline) -> pd.DataFrame:
    """
    Expects a fitted Pipeline with steps: ('preprocess', ...), ('model', LogisticRegression)
    """
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    feat_names = get_feature_names_from_preprocessor(preprocess)
    if not feat_names:
        # Best-effort fallback
        feat_names = [f"f{i}" for i in range(len(model.coef_.ravel()))]

    coefs = model.coef_.ravel()
    df = pd.DataFrame(
        {
            "feature": feat_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)

    return df
