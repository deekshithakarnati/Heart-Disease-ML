# Hyperparameter Tuning & Model Optimization

This document outlines the hyperparameter exploration, tuning methodologies, validation strategies, and experimental findings for the Random Forest Classifier.

---

## 1. Tuning Strategy & Cross-Validation Framework

To identify optimal hyperparameter configurations without data leakage:

- **Validation Technique:** Stratified $K$-Fold Cross-Validation ($k=5$ and $k=3$).
  - Ensures each fold reflects the identical $69.7\%$ healthy to $30.3\%$ heart disease proportion found in the full 9,000-record dataset.
- **Search Methodologies:**
  1. **Grid Search (`GridSearchCV`):** Exhaustive search across a predefined discrete parameter space.
  2. **Randomized Search (`RandomizedSearchCV`):** Stochastic sampling across continuous and discrete distributions for rapid initial parameter space exploration.
- **Optimization Metric:** Classification Accuracy & Macro F1-Score.

---

## 2. Hyperparameter Search Space

The following key structural and regularization hyperparameters were evaluated:

| Hyperparameter | Parameter Role | Tested Candidate Values | Selected Value |
|----------------|----------------|-------------------------|:--------------:|
| `n_estimators` | Number of individual decision trees in the ensemble | `[50, 100, 150, 200]` | **`100`** |
| `max_depth` | Maximum allowable depth for each decision tree | `[10, 15, 20, None]` | **`None`** (unconstrained) |
| `min_samples_split` | Minimum number of samples required to split an internal node | `[2, 5, 10]` | **`2`** |
| `min_samples_leaf` | Minimum number of samples required at any terminal leaf node | `[1, 2, 4]` | **`1`** (baseline) / **`2`** (regularized) |
| `max_features` | Number of features considered when looking for the best split | `['sqrt', 'log2', None]` | **`'sqrt'`** ($\approx \sqrt{25} = 5$) |
| `criterion` | Function to measure splitting quality | `['gini', 'entropy']` | **`'gini'`** |

---

## 3. Experimental Results & Parameter Comparison

### Grid Search Cross-Validation Outcome
- **Best Parameter Combination:**
  ```python
  {
      'n_estimators': 100,
      'max_depth': None,
      'min_samples_split': 2,
      'min_samples_leaf': 2
  }
  ```
- **Best Mean Cross-Validation Score (3-Fold CV):** **88.25%**
- **Test Set Accuracy:** **87.56%**

---

### Baseline vs Tuned Configuration Benchmark

| Configuration | `n_estimators` | `max_depth` | `min_samples_split` | `min_samples_leaf` | CV Accuracy | Test Accuracy | Test F1-Score |
|---------------|:--------------:|:-----------:|:-------------------:|:------------------:|:-----------:|:-------------:|:-------------:|
| **Baseline (Deployed)** | **100** | **None** | **2** | **1** | **88.10%** | **88.17%** | **0.7872** |
| **GridSearch Tuned** | 100 | None | 2 | 2 | **88.25%** | 87.56% | 0.7785 |
| **Pruned Trees** | 100 | 10 | 5 | 4 | 87.80% | 87.22% | 0.7710 |
| **Compact Ensemble** | 50 | 15 | 2 | 2 | 87.95% | 87.33% | 0.7730 |
| **Expanded Ensemble** | 200 | None | 2 | 1 | 88.15% | 88.06% | 0.7850 |

---

## 4. Hyperparameter Sensitivity Analysis

### A. Number of Trees (`n_estimators`)
- Increasing from $50 \rightarrow 100$ trees produced a noticeable reduction in ensemble variance and stabilized prediction probabilities.
- Increasing from $100 \rightarrow 200$ trees yielded diminishing returns ($+0.05\%$ CV accuracy) while doubling model memory footprint ($11.5 \text{ MB} \rightarrow 23 \text{ MB}$) and doubling inference time.
- **Conclusion:** $100$ estimators provides the optimal trade-off between predictive power and lightweight deployment.

### B. Tree Depth (`max_depth`)
- Restricting `max_depth` to $10$ or $15$ introduced mild underfitting on subtle multi-variable interactions (reducing test accuracy to $\approx 87.2\%$).
- Allowing full depth (`None`) enabled individual trees to capture intricate non-linear patterns, while ensemble averaging prevented variance inflation.

### C. Feature Subspace Size (`max_features='sqrt'`)
- Selecting $\sqrt{M} = \sqrt{25} = 5$ random features at each split proved vastly superior to considering all features (`None`).
- Subspace sampling deliberately de-correlates the 100 trees, ensuring that dominant features (like Maximum Heart Rate Achieved) do not monopolize every tree split.

---

## 5. Overfitting vs Generalization Audit

| Partition | Accuracy | Notes |
|-----------|:--------:|-------|
| **Training Set (7,200 samples)** | 99.8% | Deep decision trees fit training patterns thoroughly |
| **Cross-Validation (Mean 5-Fold)** | 88.2% | Robust out-of-fold generalization |
| **Holdout Test Set (1,800 samples)** | 88.17% | Matches CV score closely, confirming absence of data leakage |

The minimal divergence between Cross-Validation ($88.2\%$) and Holdout Testing ($88.17\%$) demonstrates strong generalization capability.
