# Credit Risk Scoring Dashboard 

An interactive **credit risk scoring** dashboard that predicts **Probability of Default (PD)** for loan applicants using a trained **Logistic Regression** pipeline.  
The app is designed like a lightweight internal tool used in lending teams: score an applicant, explore a portfolio, and review a simple “model card” for governance-style reporting.

---

## Project Overview

This project trains a baseline credit risk model on the **Statlog German Credit** dataset and serves the model through a **Streamlit** web application.

What you can do with it:
- **Underwriting view:** enter applicant details → get PD + a decision suggestion (Approve / Manual Review / Decline).
- **Portfolio view:** filter the dataset and inspect PD distributions and risk segmentation.
- **Model governance view:** inspect ROC/PR curves, calibration, confusion matrices at different thresholds, and global coefficient-based feature importance.

---

## Key Features

- **Applicant scoring** with configurable decision thresholds (Approve / Manual Review / Decline)
- **PD gauge** + **what-if simulation** (vary one feature and see PD change)
- **Portfolio explorer** with filtering and risk segmentation charts
- **Model card** page with:
  - ROC curve (from exported metrics)
  - Confusion matrix at an interactive threshold
  - Calibration curve
  - Precision–Recall curve
  - Global coefficient-based “feature importance”
- **Reproducible artifacts** tracked in `artifacts/` (model, metrics, predictions, coefficients)

---

## Installation

### 4.1 Docker Method (recommended)

**What you need installed:**
- **Docker**  
  - **Windows:** Docker Desktop with **WSL 2 enabled**  
  - **macOS:** Docker Desktop  
  - **Linux (Ubuntu):** Docker Engine or Docker Desktop

```bash
# 1) Clone the repo and navigate into it
git clone https://github.com/abhi-jith23/Credit-Risk-Dashboard.git
cd Credit-Risk-Dashboard

# 2) Build the image (run from the project root where the Dockerfile is)
docker build -t credit-risk-dashboard .

# 3) Run the container
docker run --rm -p 8501:8501 credit-risk-dashboard
````

Open the app in your browser:

* `http://localhost:8501`

---

### 4.2 Python venv Method (local run)

**Prerequisites**

* Python **3.12**
* `pip`

**Steps**

```bash
# 1) Create and activate venv
python3.12 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the Streamlit app
streamlit run app/main_app.py
```

Open:

* `http://localhost:8501`

---

### 4.3 Reproducibility notes (what was done)

* This project was developed to be fully reproducible.
* Dependencies are **pinned** in `requirements.txt`.
* The Docker image uses a fixed base image: `python:3.12.3`.
* Training uses a fixed seed (`random_state=42`) for the train/test split and model settings.
* Model outputs are exported into `artifacts/` and the Streamlit app reads these files directly.

---

## Project Structure

```text
.
├── app
│   ├── main_app.py                 # Streamlit entrypoint
│   ├── ui_text.py                  # UI text + feature labels + category mappings
│   └── pages
│       ├── applicant_scoring.py    # Underwriting / applicant PD scoring
│       ├── portfolio_explorer.py   # Portfolio filtering + segmentation
│       └── model_card.py           # Governance-style diagnostics
├── artifacts
│   ├── credit_risk_pipeline.pkl    # Trained sklearn pipeline (preprocess + model)
│   ├── metrics.json                # ROC-AUC, PR, Brier, ROC curve arrays, etc.
│   ├── test_predictions.csv        # y_true + pd_default (for interactive plots)
│   └── feature_importance.csv      # Logistic regression coefficients
├── data
│   └── german.data                 # Dataset file (used by src/data.py)
├── notebooks
│   └── train_model.ipynb           # Training notebook (calls src.train)
├── src
│   ├── data.py                     # Load/download dataset + schema
│   ├── features.py                 # Coefficient table / feature names
│   ├── metrics.py                  # Metric computation + JSON export
│   └── train.py                    # Training pipeline + artifact export
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Data Flow

### 1) Data Preprocessing

* `src/data.py` loads `data/german.data` and assigns the official feature schema.
* Numeric features are cast to numeric.
* Categorical features are kept as strings for one-hot encoding.
* Target is encoded as `default = 1` (bad credit / default) and `default = 0` (good credit).

### 2) Model Training Pipeline

Implemented in `src/train.py`:

* Split data into train/test with stratification.
* Build a preprocessing pipeline:

  * Numeric: median imputation → standardization
  * Categorical: mode imputation → one-hot encoding
* Train **Logistic Regression** with a small **GridSearchCV** over regularization settings.
* Export artifacts to `artifacts/`:

  * trained pipeline (`credit_risk_pipeline.pkl`)
  * evaluation metrics (`metrics.json`)
  * test-set probabilities (`test_predictions.csv`)
  * coefficient table (`feature_importance.csv`)

To retrain and overwrite artifacts:

```bash
python -m src.train
```

### 3) Prediction / Dashboard Pipeline

Implemented in `app/main_app.py` + pages:

* Load the trained pipeline from `artifacts/credit_risk_pipeline.pkl`.
* Load dataset (for input ranges, defaults, and portfolio exploration).
* Pages:

  * Applicant Scoring → PD + decision + what-if simulation
  * Portfolio Explorer → batch PD scoring + segmentation plots
  * Model Card → diagnostic plots and threshold analysis

---

## Disclaimer

This dashboard is an **educational demonstration** of a credit risk workflow.

* It is **not** a production-grade credit scoring system.
* Outputs are based on a historical academic dataset and a baseline model.
* Do not use this tool to make real lending decisions, credit approvals, or compliance judgments.
* Real-world credit risk systems require stronger validation, monitoring, governance, and fairness checks.

---

## Author

**Abhijith Senthilkumar**
*MSc Data Science, University of Luxembourg*

* GitHub: `https://github.com/abhi-jith23`
* Email: `abhijith.unilu@gmail.com`
