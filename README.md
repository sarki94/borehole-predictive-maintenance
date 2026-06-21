
# ML-based Predictive Maintenance for Rural Borehole Water Pumps in Northern Nigeria
### A Comparative Machine Learning Study on IoT Multi-Sensor Data

**Course:** COEN807 — Machine Learning  
**Institution:** Ahmadu Bello University, Zaria — Department of Computer Engineering  
**Track:** Supervised Learning — Multi-class Classification  

---

## Problem 


Rural communities across Northern Nigeria depend almost entirely on mechanised
borehole water pumps for potable water. Studies consistently report non-functionality
rates of 35–50% for rural water points, primarily due to reactive maintenance
practices that leave communities without water for weeks at a time.

This project applies machine learning to IoT multi-sensor pump data to predict fault
conditions **before failure occurs**, enabling proactive maintenance scheduling.

---

## Project Structure

```
borehole-predictive-maintenance/

│
├── data/                          # ← Put pump_sensor.csv here (see Download)
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis → Report §III
│   ├── 02_preprocessing.ipynb     # Cleaning & Imputation    → Report §IV (a)
│   ├── 03_feature_engineering.ipynb # Feature Engineering   → Report §IV (b)
│   ├── 04_model_training.ipynb    # Baseline Models         → Report §V
│   ├── 05_hyperparameter_tuning.ipynb # Tuning             → Report §VI
│   └── 06_evaluation.ipynb        # Results & Comparison    → Report §VII–VIII
│
├── src/
│   ├── preprocess.py              # Reusable preprocessing functions
│   ├── features.py                # Feature engineering functions
│   ├── models.py                  # Model definitions and wrappers
│   └── evaluate.py                # Metrics, plots, evaluation utilities
│
├── outputs/
│   ├── figures/                   # All saved plots (PNG, 150 dpi)
│   ├── trained_models/            # Serialised model artefacts
│   └── reports/                   # Generated summary tables
│
├── tests/
│   └── test_preprocess.py         # Unit tests for src/ modules
│
├── environment.yml                # Conda environment (recommended)
├── requirements.txt               # Pip fallback
├── .gitignore
└── README.md
```

---

## Dataset Download

The dataset is **not included** in this repository (19 MB; Kaggle terms of use).

**Option A — Browser download:**
1. Create a free account at [kaggle.com](https://kaggle.com)
2. Go to: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data
3. Click **Download** → unzip → place `pump_sensor.csv` in `data/`

**Option B — Kaggle CLI:**
```bash
pip install kaggle
# Place kaggle.json API token in ~/.kaggle/
kaggle datasets download -d nphantawee/pump-sensor-data --unzip -p data/
```

---

## Environment Setup

### Recommended: Conda (VSCode + Jupyter)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/borehole-predictive-maintenance.git
cd borehole-predictive-maintenance

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate borehole-pm

# 3. Register the environment as a Jupyter kernel
python -m ipykernel install --user --name borehole-pm --display-name "Python (borehole-pm)"

# 4. Open in VSCode
code .
# Then: Ctrl+Shift+P → "Python: Select Interpreter" → choose borehole-pm
# Open any .ipynb → select "Python (borehole-pm)" kernel in the top-right
```

### Alternative: pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name borehole-pm
```

---

## Execution Workflow

Run notebooks **in order**. Each notebook saves its outputs for the next.

| Step | Notebook | Produces |
|------|----------|---------|
| 1 | `01_eda.ipynb` | 5 figures, EDA summary |
| 2 | `02_preprocessing.ipynb` | `data/processed.pkl` |
| 3 | `03_feature_engineering.ipynb` | `data/features.pkl` |
| 4 | `04_model_training.ipynb` | Baseline model results |
| 5 | `05_hyperparameter_tuning.ipynb` | Best hyperparameters per model |
| 6 | `06_evaluation.ipynb` | Final comparison tables + figures |

```bash
# Run all notebooks non-interactively (after setup)
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb
# ... and so on
```

---

## Models Compared

| Model | Type | Library |
|-------|------|---------|
| Random Forest | Supervised Classification | scikit-learn |
| XGBoost | Supervised Classification | xgboost |
| Support Vector Machine | Supervised Classification | scikit-learn |
| Isolation Forest | Unsupervised Anomaly Detection | scikit-learn |

---

## Key Results

*(Populated after running all notebooks)*

| Model | F1-BROKEN | F1-macro | PR-AUC |
|-------|-----------|----------|--------|
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| SVM | — | — | — |
| Isolation Forest | — | — | — |

---




