# Outliers in Machine Learning — Final Complete Notes (Detection & Treatment)

*(This file consolidates **all concepts**, including the latest diagram:  
Z-score, IQR-based filtering, Percentile method, Winsorization)*

---

## 1. What are Outliers?

Outliers are observations that lie **abnormally far** from the majority of data points.

They:
- violate expected distribution
- distort statistical measures
- disproportionately impact ML models

Example:  
CGPA mostly in **6–9**, one value = **1.2 or 9.9**

---

## 2. Why Outliers Occur

- Data entry errors (extra zero, wrong unit)
- Measurement/sensor errors
- Sampling bias
- Natural rare events (fraud, defects)

👉 Treatment depends on **why** the outlier exists.

---

## 3. When Are Outliers Dangerous?

### Dangerous when:
- Dataset is small
- Algorithm is distance- or mean-based
- Model assumes Gaussian distribution
- Loss = squared error

### Not dangerous when:
- Rare events are meaningful
- Using robust or tree-based models
- Task = anomaly detection

---

## 4. Effect of Outliers on ML Algorithms

### Statistical impact

| Metric | Effect |
|------|-------|
| Mean | Highly influenced |
| Variance | Inflated |
| Std Dev | Inflated |
| Correlation | Misleading |
| Median | Robust |
| IQR | Robust |

---

### Algorithm-wise effect

- **Linear / Logistic Regression** → unstable coefficients
- **KNN** → distance distortion
- **K-Means** → centroid shift
- **PCA** → components dominated
- **Tree models** → least affected

---

## 5. How to Detect Outliers?

Detection depends on **distribution type**.

---

## 6. Techniques for Outlier Detection & Removal ✅ (As in Diagram)

---

## 6.1 Z-Score Treatment (Normal Distribution)

### Assumption
- Data is normally distributed

### Formula
\[
z = \frac{x - \mu}{\sigma}
\]

### Rule
\[
|z| > 3 \Rightarrow \text{Outlier}
\]

Based on **68–95–99.7 rule**:
- 99.7% data lies within ±3σ

✅ Simple & fast  
❌ Fails for skewed data  

Use only when distribution ≈ Gaussian.

---

## 6.2 IQR-Based Filtering ✅ (Most Used)

### Step 1: Compute
\[
IQR = Q_3 - Q_1
\]

### Step 2: Set bounds
\[
Lower = Q_1 - 1.5 \times IQR
\]
\[
Upper = Q_3 + 1.5 \times IQR
\]

Values outside bounds → outliers

✅ Robust to skew  
✅ No distribution assumption  
✅ Industry standard

---

## 6.3 Percentile Method ✅

### Idea
- Treat extreme percentiles as outliers

Example:
- Values below 1st percentile
- Values above 99th percentile

Variants:
- 5–95
- 2.5–97.5

✅ Simple  
✅ Works for skewed data  
❌ Cutoffs are arbitrary

Used heavily in finance & business datasets.

---

## 6.4 Winsorization (Capping) ✅

### Idea
- Do **not delete** outliers
- Replace with boundary thresholds

Example:
Before: 1, 2, 3, 4, 100
After : 1, 2, 3, 4, 10 (capped)


Equivalent to:
- Percentile-based capping
- IQR-based capping

✅ Preserves data size  
✅ Limits extreme influence  
✅ Better than trimming

---

## 7. Trimming (Removal)

- Completely remove outliers

✅ Simple  
❌ Data loss  
❌ Dangerous if outliers are valid

Use only when:
- Large dataset
- Confirmed noise

---

## 8. Converting Outliers to Missing Values

- Replace extreme values with `NaN`
- Apply imputation later (median / ML)

✅ Useful when data reliability is low  
❌ Needs careful justification

---

## 9. Discretization (Binning)

Convert numeric values into ranges.

Example:
90–100 → High
70–90 → Medium


✅ Reduces dominance of extremes  
✅ Useful for tree & rule-based models  
❌ Loses precision

---

## 10. Transformations

Used when outliers arise due to skewness.

- Log transform
- Square-root
- Box–Cox

✅ Compresses large values  
✅ Reduces skew  

---

## 11. Robust Scaling

Uses **median & IQR**:

\[
x' = \frac{x - Median}{IQR}
\]

✅ Resistant to outliers  
✅ Recommended before linear models

---

## 12. Model-Based Handling

Choose robust models instead of modifying data.

| Model | Robust |
|----|----|
| Random Forest | ✅ |
| XGBoost | ✅ |
| Huber Regression | ✅ |
| Quantile Regression | ✅ |

---

## 13. Outliers vs Anomalies

| Outliers | Anomalies |
|------|------|
| Statistical extremes | Behavioral extremes |
| Often noise | Often signal |
| May be removed | Must be detected |

Fraud detection focuses on **anomalies**, not removal.

---

## 14. Best Practices (Must Remember)

- Never blindly remove outliers
- Identify source first
- Normal data → Z-score
- Skewed data → IQR / Percentile
- Prefer capping over trimming
- Compare model performance before & after handling

---

## 15. Interview Golden Lines

- *Z-score works only for normal distributions.*
- *IQR is robust to skewness.*
- *Capping preserves data while controlling influence.*
- *Tree models handle outliers naturally.*
- *Outliers are dangerous only when they distort learning.*

---

## ✅ Final Summary

- Outliers are extreme deviations
- Detection methods:
  - Z-score
  - IQR
  - Percentile
- Treatment methods:
  - Trimming
  - Winsorization (capping)
  - Missing value strategy
  - Discretization
  - Transformations
  - Robust models
- No single best method — context matters
