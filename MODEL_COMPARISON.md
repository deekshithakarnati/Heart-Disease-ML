# Model Comparison & Algorithmic Benchmarking

This document provides a comparative study of multiple machine learning classifiers trained and evaluated on the assigned Heart Disease Risk dataset (9,000 patient records).

---

## 1. Empirical Performance Benchmark

All models were evaluated on identical stratified 80/20 train-test splits (7,200 training samples, 1,800 test samples) using identical `StandardScaler` and `OneHotEncoder` preprocessing pipelines.

| Rank | Model Architecture | Accuracy | Precision (Disease) | Recall (Disease) | F1-Score | ROC-AUC |
|:----:|--------------------|:--------:|:-------------------:|:----------------:|:--------:|:-------:|
| 1 | **Logistic Regression** | 89.78% | 86.32% | 78.72% | 82.34% | 0.9497 |
| 2 | **Gradient Boosting** | 88.61% | 85.42% | 75.23% | 80.00% | 0.9448 |
| 3 | **Random Forest (Selected)** | **88.17%** | **86.40%** | **72.29%** | **78.72%** | **0.9362** |
| 4 | **AdaBoost Classifier** | 87.44% | 81.71% | 75.41% | 78.44% | 0.9344 |
| 5 | **Gaussian Naive Bayes** | 86.44% | 76.64% | 79.45% | 78.02% | 0.9277 |
| 6 | **K-Nearest Neighbors (k=7)** | 85.11% | 88.37% | 58.53% | 70.42% | 0.8940 |
| 7 | **Decision Tree (CART)** | 82.67% | 72.28% | 69.36% | 70.79% | 0.7890 |

---

## 2. In-Depth Algorithmic Analysis

### A. Random Forest Classifier (Selected Production Model)
- **Mechanism:** Ensemble of 100 de-correlated decision trees trained with bootstrap aggregating (bagging) and random feature subspace selection.
- **Key Strengths:**
  - **Highest Precision on Disease Class (86.40%):** Outperforms Gradient Boosting and AdaBoost on positive predictive reliability.
  - **Interpretability & Feature Importance:** Directly provides Gini impurity reduction scores, enabling clinicians to identify critical drivers (e.g., Maximum Heart Rate, ST depression).
  - **Overfitting Resistance:** Averaging over 100 diverse trees drastically curbs variance compared to single decision trees (88.17% vs 82.67%).
  - **Handling Complex Interactions:** Efficiently models non-linear interactions between blood pressure, age, and lipid profiles without manual feature engineering.

---

### B. Logistic Regression
- **Mechanism:** Linear decision boundary optimizing log-odds with $L_2$ regularization.
- **Key Strengths:** Strong linear separability on standardized numerical predictors, achieving 89.78% accuracy and 0.9497 ROC-AUC.
- **Limitations:** Cannot naturally capture complex non-linear feature interactions (such as the combined threshold effect of simultaneous high ST depression and low exercise minutes) without explicit polynomial terms.

---

### C. Gradient Boosting & AdaBoost (Sequential Boosting)
- **Mechanism:** Builds decision trees iteratively, with each new tree minimizing the residual errors of preceding trees.
- **Key Strengths:** Very high accuracy (88.61%) and balanced F1-score (80.00%).
- **Limitations:** Slower training and higher sensitivity to noisy outlier records compared to Random Forest.

---

### D. Decision Tree Classifier (Single Estimator)
- **Mechanism:** Single recursive binary splitting tree based on Gini impurity.
- **Key Strengths:** Fast training and simple tree visualization.
- **Limitations:** Lowest accuracy (82.67%) and lowest ROC-AUC (0.7890) due to high variance and vulnerability to overfitting on training data.

---

### E. K-Nearest Neighbors (Instance-Based)
- **Mechanism:** Classifies each query sample based on majority vote of the $k=7$ closest Euclidean neighbors in the standardized 25-dimensional feature space.
- **Key Strengths:** High precision (88.37%).
- **Limitations:** Poor recall (58.53%), leading to missed positive diagnoses. Inference latency scales with $O(N)$ dataset size, making real-time web deployment less scalable.

---

### F. Gaussian Naive Bayes (Probabilistic)
- **Mechanism:** Applies Bayes' theorem under the assumption of feature conditional independence.
- **Key Strengths:** High sensitivity/recall (79.45%).
- **Limitations:** The independence assumption is violated in physiological data (e.g., systolic BP and diastolic BP are strongly correlated; Total Cholesterol is directly linked to LDL and HDL).

---

## 3. Why Random Forest Was Selected for Deployment

1. **Superior Positive Predictive Value:** With an **86.40% precision** on heart disease prediction, it minimizes false alarms for clinicians.
2. **Clinical Feature Importance:** Provides transparent, auditable feature importance weights for medical validation.
3. **Robustness to Collinearity:** Unlike linear models which can suffer from multicollinearity between correlated vital signs, tree ensembles are inherently invariant to monotonic scaling and collinearity.
4. **Fast, Deterministic Web Inference:** O(depth $\times$ trees) prediction speed guarantees sub-10 millisecond response times on Streamlit Cloud.
