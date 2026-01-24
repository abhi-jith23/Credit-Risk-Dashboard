# app/ui_text.py
APP_TITLE = "Credit Risk Scoring Dashboard"
APP_ICON = "💳"

ABOUT_SIDEBAR_MD = """
**About**
- Dataset: **German Credit (UCI Statlog)**
- Output: **PD = Probability of Default**
- Decision: thresholds → Approve / Manual Review / Decline

**Artifacts expected**
- `artifacts/credit_risk_pipeline.pkl`
- `artifacts/metrics.json`
- `artifacts/test_predictions.csv`
- `artifacts/feature_importance.csv`
"""

FEATURE_LABELS = {
    "checking_status": "Checking Account Status",
    "duration_months": "Loan Duration (months)",
    "credit_history": "Credit History",
    "purpose": "Loan Purpose",
    "credit_amount": "Credit Amount",
    "savings_status": "Savings / Bonds",
    "employment_since": "Employment Duration",
    "installment_rate": "Installment Rate (% income)",
    "personal_status_sex": "Personal Status / Sex",
    "other_debtors": "Other Debtors / Guarantors",
    "residence_since": "Years at Current Residence",
    "property": "Property",
    "age": "Age",
    "other_installment_plans": "Other Installment Plans",
    "housing": "Housing",
    "existing_credits": "Existing Credits (count)",
    "job": "Job Type",
    "num_dependents": "Number of Dependents",
    "telephone": "Telephone",
    "foreign_worker": "Foreign Worker",
}

FEATURE_HELP = {
    "duration_months": "Typical loan duration in months.",
    "credit_amount": "Total credit/loan amount.",
    "installment_rate": "Installment rate as a percentage of disposable income.",
}

# --- Official UCI code->meaning mapping (German Credit / Statlog) ---
# Source: UCI Statlog German Credit dataset documentation.  :contentReference[oaicite:4]{index=4}
CATEGORY_VALUE_LABELS = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0–200 DM",
        "A13": "≥ 200 DM / salary assignments ≥ 1 year",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits / all credits paid back duly",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly till now",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / other credits existing",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture / equipment",
        "A43": "Radio / television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100–500 DM",
        "A63": "500–1000 DM",
        "A64": "≥ 1000 DM",
        "A65": "Unknown / no savings account",
    },
    "employment_since": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1–4 years",
        "A74": "4–7 years",
        "A75": "≥ 7 years",
    },
    "personal_status_sex": {
        "A91": "Male: divorced/separated",
        "A92": "Female: divorced/separated/married",
        "A93": "Male: single",
        "A94": "Male: married/widowed",
        "A95": "Female: single",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building society / life insurance",
        "A123": "Car or other",
        "A124": "Unknown / no property",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For free",
    },
    "job": {
        "A171": "Unemployed / unskilled (non-resident)",
        "A172": "Unskilled (resident)",
        "A173": "Skilled employee / official",
        "A174": "Management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "None",
        "A192": "Yes (registered)",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

def format_category_value(feature: str, code: str) -> str:
    """Display as: Meaning - Code (while keeping the underlying value as the code)."""
    code = str(code)
    label = CATEGORY_VALUE_LABELS.get(feature, {}).get(code)
    return f"{label} - {code}" if label else code
