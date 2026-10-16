# Performance Evaluation Metrics

This document provides a rigorous mathematical and clinical evaluation of the trained Random Forest Classifier on the 1,800-patient test set ($20\%$ holdout).

---

## 1. Summary of Primary Metrics

| Metric | Score | Clinical Interpretation |
|--------|-------|-------------------------|
| **Accuracy** | **88.17%** | Overall proportion of correct predictions across all patients |
| **ROC-AUC Score** | **0.9362 (93.62%)** | Outstanding discriminative ability between healthy and diseased individuals |
| **Precision (Class 1 - Disease)** | **0.8640 (86.40%)** | 86.4% of patients flagged as at-risk genuinely had heart disease |
| **Recall / Sensitivity (Class 1)** | **0.7229 (72.29%)** | Detects 72.3% of all true heart disease cases |
| **Specificity (Class 0 - Healthy)** | **0.9506 (95.06%)** | 95.1% of healthy individuals correctly identified (very low false alarms) |
| **F1-Score (Class 1 - Disease)** | **0.7872 (78.72%)** | Harmonic mean of precision and recall for positive diagnosis |
| **Macro Average F1-Score** | **0.8548 (85.48%)** | Unweighted mean performance across both classes |
| **Weighted Average F1-Score** | **0.8804 (88.04%)** | Sample-weighted performance reflecting class balance |

---

## 2. Confusion Matrix & Mathematical Formulations

$$\begin{pmatrix}
\text{True Negatives (TN) = 1,193} & \text{False Positives (FP) = 62} \\
\text{False Negatives (FN) = 151} & \text{True Positives (TP) = 394}
\end{pmatrix}$$

### A. Accuracy
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{394 + 1193}{1800} = \frac{1587}{1800} = \mathbf{88.17\%}$$

### B. Precision (Positive Predictive Value - PPV)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{394}{394 + 62} = \frac{394}{456} = \mathbf{86.40\%}$$
- **Clinical Implication:** When the model alerts a doctor to positive cardiac risk, there is an 86.4% probability the patient truly requires intervention, avoiding unnecessary costly procedures.

### C. Sensitivity / Recall (True Positive Rate - TPR)
$$\text{Sensitivity / Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{394}{394 + 151} = \frac{394}{545} = \mathbf{72.29\%}$$
- **Clinical Implication:** The model captures nearly 3 out of every 4 heart disease cases in the population.

### D. Specificity (True Negative Rate - TNR)
$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{1193}{1193 + 62} = \frac{1193}{1255} = \mathbf{95.06\%}$$
- **Clinical Implication:** Out of 1,255 healthy individuals, 1,193 were correctly cleared, yielding an exceptionally low false alarm rate of only **4.94%**.

### E. F1-Score (Harmonic Mean)
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.8640 \times 0.7229}{0.8640 + 0.7229} = \mathbf{78.72\%}$$

---

## 3. Classification Report Breakdown

```text
              precision    recall  f1-score   support

           0       0.89      0.95      0.92      1255
           1       0.86      0.72      0.79       545

    accuracy                           0.88      1800
   macro avg       0.88      0.84      0.85      1800
weighted avg       0.88      0.88      0.88      1800
```

---

## 4. Receiver Operating Characteristic (ROC) & AUC Analysis

- **Area Under the ROC Curve (ROC-AUC):** **0.9362**
- A score above $0.90$ is categorized as **"Outstanding / Excellent Discrimination"** in biostatistical literature.
- The high AUC confirms that the model maintains robust separation between disease and non-disease distributions across virtually any operational decision threshold.

---

## 5. Clinical Precision-Recall Trade-off & Threshold Tuning

In standard classification, a default decision threshold of $\tau = 0.50$ is applied. In the web application:
- **Low Risk ($\tau < 0.30$):** High confidence of no coronary artery disease.
- **Moderate / Borderline Risk ($0.30 \le \tau < 0.50$):** Enables early preventative counseling before acute illness develops.
- **High Risk ($\tau \ge 0.50$):** Standard positive flag triggering diagnostic confirmation.
