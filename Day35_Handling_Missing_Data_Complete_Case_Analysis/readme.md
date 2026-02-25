# Missing Value Handling – Complete Expanded Notes

# 1. Introduction

Missing values appear when data for a feature is absent.  
They must be handled before using machine learning models.

There are two global strategies:
1. Removing missing values
2. Imputing missing values (filling them)

---

# 2. Removing Missing Values

## 2.1 Removing Rows
- (CCA - Complete-case analysis (CCA), also called "list-wise deletion" of cases, consists
in discarding observations(ROW) where values in any of the variables(COLUMN) are missing. Complete Case Analysis means literally analyzing only those observations for
which there is information in all of the variables in the dataset.)

+ CCA_Assumptions:
  1. Assumption_1_MCAR:
      - Meaning: "Missing Completely At Random. Missingness must not
      - Depend on the variable itself or any other variable."
      - Importance: "If MCAR does not hold, CCA becomes biased."

  1. Assumption_2_No_Distribution_Shift:
      - Meaning: "Dropping rows must not change the overall distribution of the dataset."
      - Condition: "Remaining data should represent the original population without systematic loss."

  3. Assumption_3_Sufficient_Sample_Size:
      - Meaning: "After removing incomplete rows, enough data must remain for valid analysis or model training."
      - Risk: "If too many rows are removed, statistical power and model reliability collapse."

+ Summary:
  - "CCA is valid only when missingness is MCAR."
  - "CCA must not distort the dataset’s distribution."
  - "CCA requires a large enough remaining sample size."


Remove rows only when the number of missing rows is extremely small.

```python
df.dropna()                      # drops all rows with any missing value
df.dropna(subset=['col'])        # drop rows only if 'col' is missing
df.dropna(thresh=3)              # keep rows having at least 3 non-null values
```

---

## 2.2 Removing Columns

Remove a column if:
- It has extremely high missing percentage (example > 50% or > 70%)
- It has no predictive value

```python
df.drop(columns=['col'])
```

---

# 3. Imputation of Missing Values

Imputation means replacing missing values with meaningful substitutes.

Two types exist:
- Univariate imputation (uses only the same column)
- Multivariate imputation (uses other columns to predict missing values)

---

# 4. Univariate Imputation (SimpleImputer and Manual Methods)

## 4.1 Numeric Features

### A. Mean Imputation
Best for normally distributed numeric features.

```python
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy='mean')
df['col'] = imp_mean.fit_transform(df[['col']])
```

---

### B. Median Imputation
Best for skewed numeric features or when outliers exist.

```python
imp_median = SimpleImputer(strategy='median')
df['col'] = imp_median.fit_transform(df[['col']])
```

---

### C. Random Sample Imputation
Preserves the original distribution of the feature.

```python
df['col'] = df['col'].fillna(
    df['col'].dropna().sample(df['col'].isnull().sum(), replace=True).values
)
```

---

### D. End of Distribution (EoD) Imputation
Uses extreme values so models can learn a "missing" signal.

```python
df['col'] = df['col'].fillna(df['col'].mean() + 3 * df['col'].std())
```

---

## 4.2 Categorical Features

### A. Mode Imputation
Fills missing values using the most frequent category.

```python
imp_mode = SimpleImputer(strategy='most_frequent')
df['cat'] = imp_mode.fit_transform(df[['cat']])
```

---

### B. Missing Label Category
Creates a separate category for missing values.

```python
df['cat'] = df['cat'].fillna('Missing')
```

---

# 5. Multivariate Imputation

These methods use relationships between multiple columns.

---

## 5.1 KNN Imputer

Finds nearest neighbors and uses their values for imputation.

```python
from sklearn.impute import KNNImputer

knn = KNNImputer(n_neighbors=5)
df_knn = knn.fit_transform(df)
```

---

## 5.2 Iterative Imputer (MICE)

MICE = Multiple Imputation by Chained Equations.  
Each feature with missing values is predicted using other features.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iter_imp = IterativeImputer()
df_iter = iter_imp.fit_transform(df)
```

---

# 6. Additional Advanced Notes

## 6.1 When to Choose Which Method

- Remove rows: only when missingness < 1%
- Remove columns: when feature missingness > 50% and feature is unimportant
- Mean: numeric & symmetric distribution
- Median: numeric & skewed distribution
- Mode: categorical simple fix
- Missing label: categorical with meaningful missing pattern
- Random sample: best when you want to preserve distribution
- EoD: works well with tree-based models
- KNN: small to medium datasets with correlated features
- MICE: complex datasets where performance matters

---

# 7. Summary Table

| Method              | Type         | Best For                  | Notes                                  |
| ------------------- | ------------ | --------------------------| ---------------------------------------|
| Remove Rows         | Basic        | Very few missing values   | Fast but causes data loss              |
| Remove Columns      | Basic        | >50% missing              | Only when column not important         |
| Mean                | Univariate   | Normal numeric            | Sensitive to outliers                  |
| Median              | Univariate   | Skewed numeric            | Robust to outliers                     |
| Mode                | Univariate   | Categorical               | Simple baseline                        |
| Missing Label       | Univariate   | Categorical               | Preserves missing signal               |
| Random Sample       | Univariate   | Numeric                   | Preserves variance                     |
| End of Distribution | Univariate   | Numeric                   | Helps tree models detect missingness   |
| KNN                 | Multivariate | Correlated features       | Slower on large datasets               |
| MICE                | Multivariate | Complex relations         | Most accurate advanced method          |

