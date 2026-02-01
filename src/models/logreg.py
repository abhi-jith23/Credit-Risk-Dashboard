# src/models/logreg.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class LogRegResult:
    pipeline: Pipeline
    best_params: Dict[str, Any]


def build_logreg_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> Pipeline:
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

    # scikit-learn 1.8+: penalty is deprecated; use l1_ratio instead.
    # For liblinear: l1_ratio=0 => L2, l1_ratio=1 => L1 (no elastic-net in-between).
    model = LogisticRegression(
        max_iter=4000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
        l1_ratio=0.0,  # keep explicit; safe + matches the new API direction
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train_logreg(
    X_train,
    y_train,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> LogRegResult:
    pipe = build_logreg_pipeline(numeric_cols, categorical_cols, random_state=random_state)

    C_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

    param_grid = {
        "model__C": C_grid,
        "model__l1_ratio": [0.0, 1.0],
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
    return LogRegResult(pipeline=best_pipe, best_params=search.best_params_)

