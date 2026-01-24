# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


UCI_GERMAN_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names aligned to the UCI description (20 features + target)
COLS = [
    "checking_status",         # Attribute1 (cat)
    "duration_months",         # Attribute2 (num)
    "credit_history",          # Attribute3 (cat)
    "purpose",                 # Attribute4 (cat)
    "credit_amount",           # Attribute5 (num)
    "savings_status",          # Attribute6 (cat)
    "employment_since",        # Attribute7 (cat)
    "installment_rate",        # Attribute8 (num)
    "personal_status_sex",     # Attribute9 (cat)
    "other_debtors",           # Attribute10 (cat)
    "residence_since",         # Attribute11 (num)
    "property",                # Attribute12 (cat)
    "age",                     # Attribute13 (num)
    "other_installment_plans", # Attribute14 (cat)
    "housing",                 # Attribute15 (cat)
    "existing_credits",        # Attribute16 (num)
    "job",                     # Attribute17 (cat)
    "num_dependents",          # Attribute18 (num)
    "telephone",               # Attribute19 (cat)
    "foreign_worker",          # Attribute20 (cat)
    "target_raw",              # 1=Good, 2=Bad
]

NUMERIC_COLS = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_COLS = [c for c in COLS if c not in NUMERIC_COLS and c != "target_raw"]


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    df: pd.DataFrame
    numeric_cols: list[str]
    categorical_cols: list[str]


def _ensure_downloaded(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    local_path = data_dir / "german.data"
    if not local_path.exists():
        urlretrieve(UCI_GERMAN_DATA_URL, local_path)
    return local_path


def load_german_credit(data_dir: str | Path = "data") -> DatasetBundle:
    """
    Loads the UCI Statlog German Credit dataset.
    y is encoded as: 1 = BAD (default), 0 = GOOD.
    """
    data_dir = Path(data_dir)
    local_path = _ensure_downloaded(data_dir)

    # Space-separated file, no header
    df = pd.read_csv(local_path, sep=r"\s+", header=None, names=COLS)

    # Cast numeric columns
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # Target: (1=Good, 2=Bad) -> default/bad = 1
    y = (df["target_raw"] == 2).astype(int)
    y.name = "default"

    X = df.drop(columns=["target_raw"]).copy()

    # Keep categoricals as string (good for OneHotEncoder)
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(str)

    return DatasetBundle(
        X=X,
        y=y,
        df=df.copy(),
        numeric_cols=NUMERIC_COLS.copy(),
        categorical_cols=CATEGORICAL_COLS.copy(),
    )
