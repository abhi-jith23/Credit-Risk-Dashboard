# src/train.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_german_credit
from src.features import logistic_coefficients_table
from src.metrics import compute_metrics, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
        l1_ratio=0.0,  
    )


    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return pipe


def train_and_export(
    data_dir: str | Path = PROJECT_ROOT / "data",
    artifacts_dir: str | Path = PROJECT_ROOT / "artifacts",
    random_state: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_german_credit(data_dir=data_dir)
    X, y = bundle.X, bundle.y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    pipe = build_pipeline(bundle.numeric_cols, bundle.categorical_cols)

    # Small grid (fast) – good enough for a solid baseline
    param_grid = {
        "model__C": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        "model__l1_ratio": [0.0, 1.0],  # L2 vs L1
    }


    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_pipe: Pipeline = search.best_estimator_

    # Evaluate
    proba_test = best_pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba_test, thresholds=(0.2, 0.35, 0.5))
    metrics["best_params"] = search.best_params_

    # Export pipeline + metrics
    dump(best_pipe, artifacts_dir / "credit_risk_pipeline.pkl")
    save_json(metrics, artifacts_dir / "metrics.json")

    # Export test predictions (useful later for calibration plots in Streamlit)
    pred_df = pd.DataFrame({"y_true": y_test.values, "pd_default": proba_test})
    pred_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    # Export “feature importance” (log-reg coefficients)
    coef_df = logistic_coefficients_table(best_pipe)
    coef_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    return metrics


if __name__ == "__main__":
    m = train_and_export()
    print("Training complete. Key metrics:")
    print({k: m[k] for k in ["roc_auc", "avg_precision", "brier", "default_rate", "best_params"]})
