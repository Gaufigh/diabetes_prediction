# Diabetes Risk Prediction — Team 11

Clinical risk prediction of diabetes using supervised machine learning models on patient health records (Pima Indians Diabetes dataset, 768 patients, 8 features).

**Team:** Siddhan Baranwal (230911480), Shubham Singh (230911464), Gauransh Goyal (230911456)

> This is an academic demonstration only. It is **not** a clinical diagnostic tool.

## Contents

| File | Description |
|---|---|
| `Diabetes_Risk_Prediction_Final_Complete.ipynb` | Full notebook: preprocessing, four models, test evaluation, ROC curves, confusion matrices, 5-fold stratified cross-validation, feature importance |
| `diabetes.csv` | Supplied dataset (768 rows, 8 features + `Outcome`) |
| `diabetes_model_results.csv` | Held-out test-set metrics for each model |
| `diabetes_cross_validation_results.csv` | 5-fold stratified cross-validation results on the training split |
| `diabetes_feature_importance.csv` | Random Forest feature importances |
| `Diabetes_Project_Final_Report.pdf` | Project report |

## Method

- **Cleaning:** zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are treated as missing and imputed with the training-set median.
- **Scaling:** `StandardScaler`, fitted inside each pipeline so no test-set information leaks into training.
- **Models:** Logistic Regression, Random Forest (100 trees, max depth 10), SVM with RBF kernel, and an MLP neural network (hidden layers 64 and 32, early stopping).
- **Evaluation:** accuracy, precision, recall, F1-score and ROC-AUC on a held-out test set, plus 5-fold stratified cross-validation on the training split.

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest | 0.747 | 0.660 | 0.574 | 0.614 | 0.810 |
| SVM (RBF) | 0.740 | 0.652 | 0.556 | 0.600 | 0.796 |
| MLP Neural Network | 0.721 | 0.617 | 0.537 | 0.574 | 0.818 |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |

Random Forest has the best accuracy and F1-score. The MLP has the highest test AUC-ROC.

### Cross-validation (5-fold, training split)

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.788 ± 0.019 | 0.655 ± 0.024 | 0.843 ± 0.019 |
| SVM (RBF) | 0.780 ± 0.014 | 0.648 ± 0.019 | 0.833 ± 0.023 |
| Random Forest | 0.769 ± 0.021 | 0.645 ± 0.023 | 0.824 ± 0.023 |
| MLP Neural Network | 0.770 ± 0.011 | 0.639 ± 0.026 | 0.848 ± 0.013 |

### Feature importance (Random Forest)

Glucose (0.292) is the most important feature, followed by BMI (0.161), DiabetesPedigreeFunction (0.123), Age (0.114) and Insulin (0.095).

## How to run

1. Keep the notebook and `diabetes.csv` in the same folder.
2. Install Python 3 and run:
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `Diabetes_Risk_Prediction_Final_Complete.ipynb` and choose **Run All**.

The notebook was executed against the supplied `diabetes.csv`. Numbers may vary slightly with software versions.
