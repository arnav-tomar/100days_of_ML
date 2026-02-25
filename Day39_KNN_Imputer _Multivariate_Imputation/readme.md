## KNN Imputation (K-Nearest Neighbors Imputation)

### What it does
- KNN Imputation fills a missing value using values from the **K most similar rows**.
- Similarity is computed using a distance metric (usually **Euclidean distance**) on the **available features only**.

---

### Core working idea
1. Select a row with a missing value.
2. Compute distances between this row and all other rows **that have the target value present**.
3. Ignore features that are missing in either row while computing distance.
4. Identify the **K nearest neighbors**.
5. Impute the missing value using:
   - **Mean** of neighbors (numerical)
   - **Mode** of neighbors (categorical)

---

### Distance calculation (important)
Euclidean distance with missing-aware weighting:

\[
d(x,y) = \sqrt{w \times \sum (x_i - y_i)^2}
\]

Where:
- Sum is over **only present coordinates**
-  
\[
w = \frac{\text{Total features}}{\text{Number of present features}}
\]

This penalizes rows with fewer shared features.

---

### Example (conceptual)
If a row has missing **Feature 1**:
- Use Feature 2, Feature 3, Feature 4 for distance
- Find K nearest complete rows
- Average their Feature 1 values → impute

---

### Hyperparameter: K
- Small K → low bias, high variance
- Large K → high bias, low variance
- Typical starting point: **K = 3–7**

---

### Advantages
- Preserves **local data structure**
- More accurate than mean/median imputation
- Maintains correlation between features
- Works well when data is not linearly distributed

---

### Disadvantages
- Computationally expensive (distance to all points)
- Slow for large datasets
- Sensitive to feature scaling
- Poor performance with high missingness
- Requires storing full training data

---

### When to use
- Small to medium datasets
- Strong relationship between features
- Distance-based models
- When accuracy matters more than speed

---

### When to avoid
- Very large datasets
- High-dimensional data
- Real-time inference systems
- If features are poorly correlated

---

### sklearn implementation
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = imputer.fit_transform(X)
