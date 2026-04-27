# Assignment-5-Data-1204-01
Bankruptcy Prediction with XGBoost 

Project Overview
This project focuses on building a robust machine learning pipeline to predict corporate bankruptcy. The primary goal is to move beyond simple accuracy and focus on discrimination (ranking risky companies correctly) and calibration (ensuring predicted probabilities are reliable).

The project uses XGBoost as the core model and compares it against a Logistic Regression baseline across five distinct experiments.

Dataset Summary
Source: Workshop Dataset

Target: Bankrupt? (1 = Bankrupt, 0 = Healthy)

Class Imbalance: ~3.2% Bankrupt (Positive Class)

Features: 96 financial indicators

Key Requirements Met
Stratified Split: 70% Train, 15% Validation, 15% Test.

Leakage Control: Test set was completely isolated until the final model selection.

Metrics: Prioritized PR-AUC and Brier Score over Accuracy.

AI-Assisted: Developed using Codex in VS Code to streamline evaluation functions and debugging.

Experiments Conducted
Logistic Regression: Baseline comparison with feature scaling.

XGBoost Baseline: Initial model with default hyperparameters.

XGBoost Weighted: Handling class imbalance via scale_pos_weight.

XGBoost Tuned: Manual hyperparameter optimization (Depth, Learning Rate).

XGBoost Selected Features: Using the top 20 features to test model parsimony.

How to Run
Clone this repository.

Install dependencies:

Bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
Open lastname_firstname_assignment5.ipynb in VS Code.

Select your Python kernel and click Run All.

Repository Structure
lastname_firstname_assignment5.ipynb: The complete machine learning workflow.

Results_Summary.pdf: A concise report on model performance and AI usage.

data.csv: The raw financial dataset used for training.
