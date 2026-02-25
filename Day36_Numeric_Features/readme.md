# 4. Univariate Imputation (Numeric Features)

Univariate imputation fills missing values in one column at a time using simple statistical or rule-based methods.  
Used when missingness is random or when features can be treated independently.

---

## 4.1 Numeric Features

---

## A. Mean Imputation  
Best for normally distributed numeric features.

### When to Use
- Distribution is symmetric or close to normal.  
- No heavy outliers.  
- Quick baseline imputation.

### When Not to Use
- Skewed distributions.  
- Strong outliers.

### Code
```python
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy='mean')
df['col'] = imp_mean.fit_transform(df[['col']])
```

```python
df['col'] = df['col'].fillna(df['col'].mean())
```

---

## B. Median Imputation  
More robust than mean. Works with skewed data and outliers.

### When to Use
- Skewed numeric features.  
- Outliers present.

### When Not to Use
- Distribution is symmetric (mean is better).

### Code
```python
from sklearn.impute import SimpleImputer

imp_median = SimpleImputer(strategy='median')
df['col'] = imp_median.fit_transform(df[['col']])
```

```python
df['col'] = df['col'].fillna(df['col'].median())
```

---

## C. Random Sample Imputation  
Preserves the original distribution by sampling from non-missing values.

### When to Use
- Maintain variance and distribution.  
- Avoid artificial values.

### When Not to Use
- Very small datasets.

### Code
```python
df['col'] = df['col'].fillna(
    df['col'].dropna().sample(
        df['col'].isnull().sum(),
        replace=True
    ).values
)
```

---

## D. End of Distribution (EoD) Imputation  
Creates an outlier-like value so the model detects missingness.

### When to Use
- Tree-based models (XGBoost, RandomForest, LightGBM).  
- You want missingness to carry a signal.

### When Not to Use
- Linear regression.  
- Long-tailed distributions.

### Code
```python
df['col'] = df['col'].fillna(
    df['col'].mean() + 3 * df['col'].std()
)
```

---

## E. Arbitrary Value Imputation  
Fills missing values with a fixed constant (−999, 9999, 0, etc.).  
Works only when the chosen value does not occur naturally in the feature.

### When to Use
- Want the model to easily detect missingness.  
- Value is guaranteed to be outside the feature's natural range.  
- Ideal for tree models.

### When Not to Use
- Arbitrary value falls within normal feature range.  
- Linear models where extreme values cause distortion.

### Recommended Constants
- −999  
- −1  
- 999  
- Context-specific special numbers  

### Code
```python
arbitrary_value = -999
df['col'] = df['col'].fillna(arbitrary_value)
```

```python
from sklearn.impute import SimpleImputer

imp_arbitrary = SimpleImputer(strategy='constant', fill_value=-999)
df['col'] = imp_arbitrary.fit_transform(df[['col']])
```

---

## Note
Univariate methods ignore relationships between columns.  
For better accuracy, use multivariate methods (KNNImputer, IterativeImputer/MICE) when missingness depends on other features.
