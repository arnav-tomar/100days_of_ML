
# 📘 Linear Regression Assumptions

---

# 🧠 Why Assumptions Matter
Linear Regression is not just a formula.
---

# ⭐ The 5 CORE Assumptions

---

# 1️⃣ LINEARITY

## 📌 Definition
There must be a **linear relationship** between:
- Independent variables (X)
- Dependent variable (y)

Mathematically:
y = β₀ + β₁x₁ + β₂x₂ + ε

---

## 👀 Visual Intuition

### ✅ Valid
- Straight upward trend
- Straight downward trend
- Slight noise allowed

### ❌ Invalid
- Curves
- Exponential growth
- Saturation patterns

---

## 🧪 How to Check

### Scatter Plots (Most Important)
```python
import seaborn as sns
import matplotlib.pyplot as plt

for col in X.columns:
    sns.scatterplot(x=X[col], y=y)
    plt.title(f"{col} vs Target")
    plt.show()
```

---

## 🚨 If Violated
Fix using:
- Polynomial regression
- Log transformation
- Non-linear models (Tree, NN)

---

# 2️⃣ NO MULTICOLLINEARITY

## 📌 Definition
Independent variables must be:
➡ **NOT strongly correlated with each other**

Bad example:
- Height & Weight (often correlated)
- Experience & Age

---

## 💣 Why It's Dangerous
Multicollinearity causes:
- Unstable β coefficients
- Inflated variance
- Hard interpretation
- Feature importance confusion

---

## 🧠 Intuition
If two features carry same info → model gets confused:
“Who actually caused the change?”

---

## 🔍 Detection Methods

### 1️⃣ Correlation Heatmap
```python
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
```

### Rule:
| Correlation | Meaning |
|------------|--------|
| < 0.5 | Safe |
| 0.5–0.8 | Moderate |
| > 0.8 | Risky |

---

### 2️⃣ VIF (Most Important)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
```

### 🎯 VIF Interpretation
| VIF | Meaning |
|-----|--------|
| 1 | Perfect |
| <5 | OK |
| 5–10 | Warning |
| >10 | Remove feature |

---

## 🛠 Fix Multicollinearity
- Drop features
- PCA
- Ridge regression

---

# 3️⃣ NORMALITY OF RESIDUALS

## 📌 Residual = Error
Residual = Actual − Predicted

ε = y − ŷ

---

## 📌 Assumption
Residuals should follow:
➡ **Normal Distribution (Gaussian)**

---

## 🤔 Why Needed?
Important for:
- Confidence intervals
- p-values
- Hypothesis testing

NOTE:
Not very important for pure prediction.

---

## 🔍 Visual Checks

### Histogram
```python
sns.histplot(residuals, kde=True)
```

### Q-Q Plot (Most Reliable)
```python
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()
```

### ✅ Ideal Q-Q Plot
Points lie on diagonal line.

---

## 🛠 Fix If Violated
- Log transform target
- Remove outliers
- Robust regression

---

# 4️⃣ HOMOSCEDASTICITY

## 📌 Definition
Residual variance should be:
➡ **Constant across predictions**

Fancy word:
Homoscedasticity = Equal spread

Opposite:
Heteroscedasticity = Unequal spread

---

## 👀 Visual Intuition

### ✅ Good
Random cloud

### ❌ Bad
- Funnel shape
- Cone pattern
- Increasing variance

---

## 🔍 Test Method

### Residual vs Prediction Plot
```python
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()
```

---

## 🚨 Why It Matters
Violations cause:
- Wrong confidence intervals
- Biased standard errors
- Invalid inference

---

## 🛠 Fix Methods
- Log transformation
- Weighted regression
- Robust regression

---

# 5️⃣ NO AUTOCORRELATION

## 📌 Definition
Residuals should be:
➡ Independent of each other

Important for:
- Time series
- Sequential data

---

## ❌ Bad Example
Error today depends on error yesterday.

This breaks independence.

---

## 🔍 Detection Methods

### Visual
```python
plt.plot(residuals)
plt.title("Residual Sequence")
```

Random pattern = Good

---

### Statistical Test — Durbin Watson
```python
from statsmodels.stats.stattools import durbin_watson
durbin_watson(residuals)
```

### 🎯 Interpretation
| Value | Meaning |
|------|--------|
| ~2 | No autocorrelation |
| <1.5 | Positive autocorr |
| >2.5 | Negative autocorr |

---

## 🛠 Fix Methods
- Add lag features
- Time series models (ARIMA)
- GLS regression

---

# 🧪 Residual Workflow (COLAB READY)

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
residuals = y_test - y_pred
```

Now run all diagnostics.

---

# 🎯 INTERVIEW CHEAT SHEET

## 🔥 MUST REMEMBER ORDER
1️⃣ Linearity  
2️⃣ No multicollinearity  
3️⃣ Normal residuals  
4️⃣ Homoscedasticity  
5️⃣ No autocorrelation  

---

# 🧠 Smart Interview Tips

### Q: Most important assumption?
Depends:
- Inference → Normality
- Prediction → Linearity

---

### Q: Most commonly violated?
- Multicollinearity
- Heteroscedasticity

---

### Q: If assumptions fail?
Say:
“Use transformation or switch model.”

Instant + points in interview.

---

# 🧭 Real World Perspective

## In Statistics
Assumptions = CRITICAL

## In Machine Learning
Assumptions = Less strict
But still useful for:
- Explainability
- Feature importance
- Business insights

---
---
