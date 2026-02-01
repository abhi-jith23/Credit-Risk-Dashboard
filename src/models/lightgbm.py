# src/models/lightgbm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class LGBMResult:
    pipeline: Pipeline
    best_params: Dict[str, Any]


def build_lgbm_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Optional but often helps small datasets by reducing sparse explosion:
            # merge rare categories into an "infrequent" bucket
            ("onehot", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                sparse_output=True
            )),
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

    preprocess.set_output(transform="default")

    model = LGBMClassifier(
        random_state=random_state,
        n_jobs=1,                 # avoid nested parallelism
        class_weight="balanced",
        objective="binary",
        verbosity=-1,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train_lightgbm(
    X_train,
    y_train,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> LGBMResult:
    pipe = build_lgbm_pipeline(numeric_cols, categorical_cols, random_state=random_state)

    # Bigger search space (regularization + subsampling are key)
    param_distributions = {
        "model__n_estimators": [200, 400, 800, 1200],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63, 127],
        "model__max_depth": [-1, 3, 4, 5, 6],
        "model__min_child_samples": [10, 20, 30, 50, 80],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
        "model__min_split_gain": [0.0, 0.01, 0.05],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=30,               # 30 * 5 = 150 fits; good tradeoff
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        random_state=random_state,
    )

    search.fit(X_train, y_train)
    best_pipe: Pipeline = search.best_estimator_
    return LGBMResult(pipeline=best_pipe, best_params=search.best_params_)
