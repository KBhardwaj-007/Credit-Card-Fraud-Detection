# Credit Card Fraud Detection: An End-to-End MLOps Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KBhardwaj-007/Credit-Card-Fraud-Detection/blob/main/Credit_Card_Fraud_Detection.ipynb)

This repository contains a comprehensive, production-ready MLOps project for detecting fraudulent credit card transactions in real-time. The project goes beyond simple model training to build a robust, explainable, and maintainable system, culminating in a multi-page interactive dashboard built with Streamlit.

The core of the system is a **Hybrid Ensemble Model** that combines supervised and unsupervised learning techniques, optimized not for accuracy, but for minimizing real-world **business cost**.

---

## Key Features: MLOps in Action

This project implements a full suite of MLOps best practices to create a system that is not only powerful but also transparent, reliable, and ready for a production environment.

| Feature | Description |
| :--- | :--- |
| **Hybrid Ensemble Model** | Combines the strengths of a **Random Forest**, a **Cost-Sensitive Neural Network (FCNN)**, and an **unsupervised Autoencoder** to maximize detection robustness and accuracy. |
| **Cost-Sensitive Learning** | The model is explicitly optimized to minimize a business-defined cost function (e.g., \$100 for a missed fraud, \$5 for a false alarm), directly aligning its performance with financial objectives. |
| **Model Explainability (XAI)** | Integrates **SHAP** and **LIME** to provide clear, human-readable explanations for every prediction, answering the critical question: "*Why was this transaction flagged?*" |
| **Automated Fraud Reporting** | Automatically generates auditable **PDF reports** for high-risk transactions, complete with transaction details and a SHAP explanation. Essential for compliance and fraud analysis teams. |
| **Real-Time Monitoring Dashboard** | A multi-page Streamlit application that includes a live analytics page, simulating a real-time transaction feed with fraud alerts and performance tracking. |
| **Data Drift Detection** | Proactively monitors incoming data for changes in distribution using the **Population Stability Index (PSI)**. The system raises alerts when the model may be becoming stale, triggering a need for review or retraining. |

---

## Demo

The project is deployed as a multi-page Streamlit dashboard.
The dashboard includes:
1.  **Prediction Page:** Make real-time predictions on single transactions and get instant model explanations.
2.  **Model Comparison Page:** A visual, head-to-head comparison of all trained models, justifying the choice of the final Hybrid Ensemble.
3.  **Analytics Page:** A real-time monitoring dashboard with live fraud alerts and data drift detection.

---

## Project Structure

The repository is organized to separate configuration, source code, data, and outputs, following modern data science project standards.

```
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Main Jupyter Notebook for development and experimentation
├── README.md                           # This README file
├── requirements.txt                     # Project dependencies
│
├── data/
│   ├── raw/                             # For the original creditcard.csv dataset
│   └── processed/                       # For scaled and split data (X_train.pkl, etc.)
│
├── models/                             # For saved model files (.pkl, .keras) and scalers
│
├── reports/
│   ├── figures/                        # For plots and charts generated during analysis
│   └── fraud_reports/                   # For auto-generated PDF fraud reports
│
├── app.py                             # Main entry point for the Streamlit app
├── pages/                             # Contains the individual pages of the Streamlit app
│   ├── prediction_1.py
│   ├── model_comparison_2.py
│   └── analytics_3.py
│
└── src/                                # Source code for the project
    ├── __init__.py
    ├── config.py                        # Central configuration file
    ├── preprocessing.py                 # Data scaling and splitting pipeline
    ├── metrics.py                       # Custom evaluation metrics and plotting
    ├── models.py                       # Baseline model training logic
    ├── nn_models.py                      # FCNN architecture
    ├── training.py                     # Neural network training loop
    ├── autoencoder.py                    # Autoencoder architecture and logic
    ├── explainability.py               # SHAP/LIME integration and PDF reporting
    └── utils.py                        # Core utilities for the Streamlit app
```

---

## Getting Started

Follow these steps to set up and run the project in a local or cloud environment like Google Colab.

### Prerequisites

*   Git
*   Python >= 3.10
*   An environment manager (like `conda` or `venv`) is recommended.

### 1. Clone the Repository

```bash
git clone https://github.com/KBhardwaj-007/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Set Up the Environment

Create and activate a new virtual environment, then install the required dependencies.

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the Dataset

The dataset is not included in this repository. Follow the instructions in `data/raw/placeholder_instructions.txt` to download it:
1.  Go to the [Kaggle Credit Card Fraud Detection Dataset page](https://www.kaggle.com/mlg-ulb/creditcardfraud).
2.  Download the `creditcard.csv` file.
3.  Place the `creditcard.csv` file inside the `data/raw/` directory.

### 4. Run the Main Notebook

The `Credit_Card_Fraud_Detection.ipynb` notebook contains the complete end-to-end workflow. Open it in a Jupyter environment and run the cells sequentially to perform:
1.  **Data Preprocessing:** This will generate the scaled and split data files in `data/processed/`.
2.  **Model Training:** This will train all models and save the final artifacts (e.g., `scaler.pkl`, `random_forest.pkl`, `FCNN_final.keras`) to the `models/` directory.

---

## Usage: Running the Streamlit Dashboard

Once the main notebook has been run and the model artifacts have been saved, you can launch the interactive dashboard.

From the root directory of the project, run the following command in your terminal:

```bash
streamlit run app.py
```

This will launch the multi-page application in your web browser, where you can interact with the prediction, comparison, and analytics pages.

---

## Final Model Performance

The final **Hybrid Ensemble Model** was optimized for business cost and demonstrates state-of-the-art performance on the unseen test set.

| Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **Total Business Cost** | **$1,255** | The lowest cost achieved, balancing fraud losses and operational overhead. |
| **PR-AUC** | **0.867** | Excellent performance on the imbalanced dataset (Target: > 0.80). |
| **Recall** | **0.8878** | **Catches ~89% of all true fraud cases** (Target: > 0.85). |
| **Precision** | **0.7373** | When the model flags a transaction, it is correct ~74% of the time. |
| **Missed Frauds (FN)** | **11** | An extremely low number of high-cost errors. |
| **False Alarms (FP)** | **31** | An operationally efficient low number of false alerts. |

---

## Business Insights

This project yielded several key insights that translate directly to business value:

1.  **Cost-Optimization is King:** Optimizing for a business-specific cost function delivered a **73% cost reduction** compared to a standard, un-tuned deep learning model. Accuracy is not the right metric for this problem.
2.  **Transparency Drives Efficiency:** Explainability tools like SHAP and LIME provide immediate, actionable reasons for fraud alerts, enabling analysts to **reduce investigation time** and build trust in the system.
3.  **Proactive Monitoring Prevents Failure:** Data drift detection acts as an **early warning system**, allowing the business to retrain the model *before* its performance degrades and fraud losses increase.

---

## Acknowledgments

This project uses the "Credit Card Fraud Detection" dataset from Kaggle, which was collected and anonymized by:
*   Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
*   Published in *Calibrating Probability with Undersampling for Unbalanced Classification*. In *Symposium on Computational Intelligence and Data Mining (CIDM), IEEE*, 2015.
