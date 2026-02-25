digitstrain.csv :- https://www.kaggle.com/datasets/richardgarciav/digit-train-csv

# 🌲 Random Forest Hyperparameters
### Classification + Regression
---

## 1. High-level Hyperparameters (Forest-level)

These control the **ensemble itself**.

### 1.1 `n_estimators`
Number of trees in the forest.

$$
\text{Final prediction} =
\begin{cases}
\text{majority vote} & \text{(classification)} \\\\
\text{mean prediction} & \text{(regression)}
\end{cases}
$$

- Low value → high variance
- High value → stable, but slower
- Typical range: **$100$ – $500$**

---

### 1.2 `max_features`
Number of features considered **at each split**.

Let total features = $d$

| Value | Meaning |
|-----|--------|
| `"sqrt"` | $\sqrt{d}$ (default for classification) |
| `"log2"` | $\log_2 d$ |
| `None` | all features |
| float $p$ | $p \times d$ |
| int $k$ | exactly $k$ features |

📌 **Key idea**:  
Random Forest does **node-level feature sampling**, not tree-level.

This increases **decorrelation** between trees.

---

### 1.3 `bootstrap`
Whether sampling is done **with replacement**.

- `True` → classic Random Forest
- `False` → uses full data per tree (less randomness)

---

### 1.4 `max_samples`
Fraction or number of rows per tree.

If dataset size = $N$

- float $p$ → $pN$ rows
- int $k$ → $k$ rows

📌 Best range: **$0.5$ – $0.75$**

Too small → high bias  
Too large → higher variance

---

## 2. Tree-level Hyperparameters (Bias–Variance control)

These control **each decision tree**.

---

### 2.1 `max_depth`
Maximum depth of each tree.

- Small → underfitting (high bias)
- Large → overfitting (high variance)

📌 RF works best with **deep but noisy trees**

---

### 2.2 `min_samples_split`
Minimum samples required to split a node.

$$
n_{node} \ge \text{min\_samples\_split}
$$

- Larger → smoother boundary
- Smaller → aggressive splits

---

### 2.3 `min_samples_leaf`
Minimum samples in a leaf.

Ensures prediction stability:

$$
\hat{y}_{leaf} = \frac{1}{n} \sum y_i
$$

- Prevents tiny noisy leaves
- Very important for regression

---

### 2.4 `max_leaf_nodes`
Hard limit on number of leaves.

Directly controls model complexity.

---

### 2.5 `min_impurity_decrease`
Split happens only if:

$$
\Delta I \ge \text{threshold}
$$

Used for **early stopping**.

---

## 3. Split Criterion (Loss function)

### Classification:
- `"gini"`:
$$
G = 1 - \sum p_k^2
$$

- `"entropy"`:
$$
H = -\sum p_k \log p_k
$$

### Regression:
- `"squared_error"`:
$$
\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2
$$
- `"absolute_error"` (robust to outliers)

---

## 4. Sampling & Class Control

### 4.1 `class_weight`
Used for **imbalanced datasets**.

$$
\text{weighted loss} = w_c \cdot \text{loss}
$$

Options:
- `None`
- `"balanced"`
- `{class: weight}`

---

## 5. Performance & Reproducibility

### 5.1 `n_jobs`
Parallelism.

- `-1` → use all CPU cores

---

### 5.2 `random_state`
Controls randomness for:
- row sampling
- feature sampling

Ensures reproducibility.

---

### 5.3 `verbose`
Controls training logs.

---

### 5.4 `warm_start`
Allows incremental training:

$$
n_{\text{trees}}^{new} = n_{\text{old}} + \Delta
$$

---

## 6. Cost-Complexity Pruning

### `ccp_alpha`
Post-pruning strength.

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

- $\alpha = 0$ → no pruning
- Higher $\alpha$ → simpler trees

---

## 7. Classification vs Regression Differences

| Parameter | Classification | Regression |
|--------|--------------|-----------|
| criterion | gini / entropy | mse / mae |
| voting | majority | mean |
| bias sensitivity | lower | higher |

Everything else is identical.

---

## 8. Recommended Defaults (Real-world)

```python
RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    max_samples=0.7,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
9. Interview One-liners

Random Forest reduces variance, not bias

Uses node-level feature sampling

Deep trees + averaging = stability

More randomness → better generalization

Better than bagging due to extra decorrelation

Final Takeaway

Random Forest works because it converts:

Low-bias, high-variance trees
into
Low-bias, low-variance ensemble

using bootstrapping + feature randomness + aggregation.
