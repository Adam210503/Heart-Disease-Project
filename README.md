# 🫀 CardioRisk: Predictive Cardiac Disease Severity Assessment

A full-stack, production-ready machine learning system for multi-class cardiac disease risk stratification, built on the UCI Cleveland Heart Disease dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

CardioRisk is an end-to-end machine learning pipeline that predicts the **severity class** (0–4) of coronary artery disease from 13 standard clinical features. The project goes beyond a basic classifier that pairs a calibrated, SMOTE-balanced Random Forest with a clinical-grade Streamlit dashboard and a REST API backend, demonstrating how a model is taken from a research notebook to a deployable decision-support tool.

The system is split into three layers:

| Layer | File | Purpose |
|---|---|---|
| Model training | `train_model.py` | Data pipeline, model selection, tuning, calibration, serialisation |
| REST API | `main.py` | FastAPI scoring endpoint that serves the trained model over HTTP |
| Dashboard | `app.py` | Streamlit clinical interface that calls the API and renders results |

---

## Key ML Engineering Decisions

This section explains the non-trivial choices made during development.

### Multi-class framing (5 severity levels, non-binary)
The Cleveland dataset's target variable encodes five severity classes (0 = no disease, 1–4 = increasing coronary artery narrowing). Most tutorials collapse this to binary (disease / no disease). Retaining all five classes makes the problem substantially challenging and more realistic, requiring proper handling of class imbalance and appropriate choice of evaluation metric.

### SMOTE with minority-class-aware `k_neighbors`
Class 4 (critical disease) contains only 13 samples. Standard SMOTE with the default `k_neighbors=5` would fail because a 5-fold split of 13 samples leaves fewer than 5 neighbours available. k_neighbors is restricted to 2 (default is 5) because minority classes (e.g., Severity Class 4) have extremely low sample sizes in the training split. Lowering this prevents ValueError crashes during synthetic sample generation.

### Macro F1 as the primary optimisation metric
Accuracy is a misleading metric on imbalanced data — a model that always predicts class 0 achieves ~54% accuracy on this dataset and learns nothing. Hence, `GridSearchCV` is configured with `scoring='f1_macro'`, which weights each class equally regardless of its frequency and penalises the model for ignoring minority classes.

### Calibrated probability outputs
`RandomForestClassifier` produces scores, not true probabilities. The model is wrapped in `CalibratedClassifierCV` with sigmoid calibration, so that a displayed score of 72% genuinely means that approximately 72% of patients with this clinical profile had disease at that severity level. This is critical for clinical credibility.

### Decoupled API + frontend architecture
Rather than loading the model directly in the Streamlit app, the dashboard calls a FastAPI backend over HTTP. The model server can be scaled, versioned, or swapped independently of the UI layer.

---

## Model Performance

Training used stratified 5-fold cross-validation. Results on the SMOTE-balanced training set:

### 📊 Model Performance Metrics

Following hyperparameter tuning and class balancing via SMOTE, the final model achieved an overall cross-validated **accuracy of 89%**. 

| Severity Target | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0** *(No Disease)* | 0.86 | 0.90 | 0.88 | 164 |
| **Class 1** *(Mild)* | 0.80 | 0.77 | 0.79 | 164 |
| **Class 2** *(Moderate)* | 0.93 | 0.92 | 0.93 | 164 |
| **Class 3** *(Severe)* | 0.90 | 0.91 | 0.91 | 164 |
| **Class 4** *(Critical)* | 0.96 | 0.96 | 0.96 | 164 |
| **Accuracy** | — | — | **0.89** | **820** |
| **Macro Average** | 0.89 | 0.89 | 0.89 | 820 |
| **Weighted Average** | 0.89 | 0.89 | 0.89 | 820 |

**Final model**: Calibrated Random Forest, hyperparameters selected via GridSearch over `n_estimators`, `max_depth`, and `min_samples_split`, trained on SMOTE-balanced data.

---

## Clinical Features

| Feature | Description | Type |
|---|---|---|
| `age` | Patient age in years | Continuous |
| `sex` | Biological sex (0 = Female, 1 = Male) | Binary |
| `cp` | Chest pain type (1 = Typical Angina → 4 = No chest pain) | Categorical |
| `trestbps` | Resting blood pressure (mmHg) | Continuous |
| `chol` | Serum cholesterol (mg/dL) | Continuous |
| `fbs` | Fasting blood sugar > 120 mg/dL | Binary |
| `restecg` | Resting ECG result (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy) | Categorical |
| `thalach` | Maximum heart rate achieved during stress test (bpm) | Continuous |
| `exang` | Exercise-induced angina (0 = No, 1 = Yes) | Binary |
| `oldpeak` | ST depression induced by exercise relative to rest (mm) | Continuous |
| `slope` | Slope of peak exercise ST segment (1 = Upsloping, 2 = Flat, 3 = Downsloping) | Categorical |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) | Ordinal |
| `thal` | Thallium stress test result (3 = Normal, 6 = Fixed defect, 7 = Reversible defect) | Categorical |

**Target variable**: Severity class 0 (no disease) through 4 (critical multi-vessel disease).

---

## Project Structure

```
cardiorisk/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                            # FastAPI REST scoring API gateway
│   │   └── models/
│   │       └── patient.py                     # Pydantic request schema
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                                 # Streamlit clinical dashboard UI
│   ├── Dockerfile
│   └── requirements.txt
├── ml/
│   ├── train.py                               # Full model training, calibration & validation pipeline
│   ├── data/
│   └── models/
│       └── optimized_heart_disease_model.pkl  # Serialized ML pipeline artifact (generated by train.py)
├── docker-compose.yml                         # Multi-container orchestration & local networking layout
└── README.md
```

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Adam210503/CardioRisk.git
cd cardiorisk
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### Step 1 - Train the model

This downloads the Cleveland dataset, runs cross-validation, performs hyperparameter tuning, applies SMOTE balancing, calibrates probabilities, and saves the model pipeline.

```bash
python ml/train.py
```

Expected output: `ml/models/optimized_heart_disease_model.pkl`, plus a printed cross-validation table and classification report.

### Step 2 - Start the FastAPI backend

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can inspect the auto-generated docs at `http://127.0.0.1:8000/docs`.

### Step 3 - Launch the Streamlit dashboard

Open a second terminal tab:

```bash
streamlit run frontend/app.py
```

The dashboard will open at `http://localhost:8501`. Configure patient metrics in the sidebar and click **Analyse Cardiac Risk Profile**.

---

## API Reference

### `GET /`
Health check endpoint.

```json
{ "status": "online", "message": "CardioRisk API is operational." }
```

### `POST /predict`
Accepts a JSON patient record and returns the predicted severity class and class probabilities.

**Request body:**
```json
{
  "age": 54.0,
  "sex": 1.0,
  "cp": 4.0,
  "trestbps": 140.0,
  "chol": 239.0,
  "fbs": 0.0,
  "restecg": 1.0,
  "thalach": 160.0,
  "exang": 0.0,
  "oldpeak": 1.2,
  "slope": 2.0,
  "ca": 0.0,
  "thal": 7.0
}
```

**Response:**
```json
{
  "severity_class": 2,
  "probabilities": [0.12, 0.18, 0.41, 0.19, 0.10]
}
```

`severity_class` maps to: 0 = No disease, 1 = Mild narrowing, 2 = Moderate narrowing, 3 = Severe disease, 4 = Critical disease.

---

## 📦 System Requirements & Dependencies

Dependencies are split per service. Install only what each component needs:

```bash
pip install -r backend/requirements.txt   # FastAPI, scikit-learn, joblib
pip install -r frontend/requirements.txt  # Streamlit, requests, pandas, numpy

---
## Containerized Local Development Workflow

The application is fully containerized and orchestrated using Docker and Docker Compose, eliminating the need for manual, multi-terminal service instantiation.

### Prerequisites
* Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and initialized on the host machine.

### Execution
To build the target images, configure the isolated multi-container network infrastructure, and initialize both services simultaneously, run the following command from the project root:

```bash
docker-compose up --build

---

## Limitations & Model Card

> ⚠️ Built for portfolio and educational purposes only. Not intended for clinical decision-making.

- **Small dataset.** 303 samples is limited by modern ML standards; generalisation to broader populations is unverified.
- **Minority class sparsity.** Class 4 has only 13 real samples. SMOTE compensates synthetically, but predictions at critical severity carry higher uncertainty.
- **Historical data.** Collected at a single institution in the 1980s. Does not represent contemporary global patient demographics.
- **Legacy encoding.** The `thal` feature uses a categorical encoding inconsistent with modern nuclear cardiology reporting.
- **No external validation.** The model has not been tested against prospective clinical cohorts or external datasets.

---

## Learning Takeaways

- Handling class imbalance in multi-class problems with SMOTE, including the `k_neighbors` constraint imposed by small minority classes
- Why accuracy fails on imbalanced data and how macro F1 corrects for it
- The difference between raw model scores and calibrated probabilities, and why calibration matters in risk-sensitive contexts
- Architecting a decoupled ML system with a separate API layer and a frontend that communicates with it over HTTP
- Using sklearn's `Pipeline` to bundle preprocessing and inference into a single serialisable object

---

## Author

**Adam Mikail**  
[LinkedIn](https://www.linkedin.com/in/adammikail/) · [Email](mailto:adammikail2105@gmail.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

