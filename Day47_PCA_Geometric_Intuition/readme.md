# Principal Component Analysis (PCA) – Intuition, Math, and Usage

## 0. Big Picture

When your dataset has **too many features (columns)**:

- Training becomes slow.
- Models overfit more easily.
- Distances in high dimensions become weird (curse of dimensionality).
- Visualizing data is impossible beyond 3D.

To fix this, we use **dimensionality reduction**.

There are two main strategies:

1. **Feature Selection** – keep *some* of the original features, drop the rest.  
2. **Feature Extraction** – create *new* features from the original ones (combinations / transformations).

👉 **PCA (Principal Component Analysis)** is a **feature extraction** technique for **unsupervised** dimensionality reduction.

- Input: only **X (features)**, no y (labels).
- Output: a new set of features called **principal components**.
- Goal: **reduce dimensions while preserving as much information (variance) as possible**.

---

## 1. Curse of Dimensionality (Quick Recap)

Suppose you keep adding more and more features:

- Up to some point, new features help your model.
- After that point, adding more noisy / redundant features:
  - does **not** improve performance,
  - may **reduce** performance,
  - increases computational cost,
  - increases risk of overfitting.

So there is an **optimal number of features**, beyond which you only get **cost but no benefit**.

Dimensionality reduction tries to:

- Remove useless/redundant features.
- Or compress them into fewer, more informative features.

---

## 2. Feature Selection vs Feature Extraction

### 2.1 Feature Selection (Keep a Subset of Original Features)

You **choose some original columns** and drop others.

Example dataset:

- `rooms`  – number of rooms in a house
- `grocery_shops` – number of grocery shops near the house
- `price`  – price of the house (target)

Intuitively:

- `rooms` clearly affects `price` (more rooms → higher price).
- `grocery_shops` might matter a bit, but not as strongly.

So if you must keep **only one** feature:
- You will keep **`rooms`**, and drop **`grocery_shops`**.

This is **feature selection**: choose `rooms`, drop `grocery_shops`.

#### Geometric intuition for feature selection

Plot a scatter:

- x-axis: `rooms`
- y-axis: `grocery_shops`

You project all points onto each axis:

- Projection on `rooms` → spread is **large**.
- Projection on `grocery_shops` → spread is **small**.

The axis with **larger spread (variance)** carries more information about how data points differ.

So you **keep the feature with higher variance** → `rooms`.

That’s essentially what many feature selection methods do:
- Prefer features with high variance / stronger relationship with the target.

### 2.2 Where Feature Selection Fails

Change the second feature:

- `rooms`
- `bathrooms`
- `price`

Now:

- `price` depends on **both** `rooms` and `bathrooms`.
- If you drop either `rooms` or `bathrooms`, you lose important information.

Geometrically:

- If you plot `rooms` vs `bathrooms`, the points lie roughly along a **diagonal line**:
  - houses with more rooms usually have more bathrooms.
- The spread along `rooms` and along `bathrooms` axes is now **similar**.
- So simple variance comparison cannot clearly say “keep only rooms” or “keep only bathrooms”.

Feature selection here is awkward: both features matter and are correlated.

You need something smarter.

---

## 3. Feature Extraction – The Core Idea

Instead of choosing one of `{rooms, bathrooms}`, you can define a **new feature**:

- `size_of_flat = some_function(rooms, bathrooms)`

Example conceptual idea:

- More rooms + more bathrooms → larger flat.
- So `size_of_flat` summarizes both.

Now your dataset is:

- `size_of_flat`
- `price`

You replaced two correlated features with **one combined feature** that still carries the important information.

This is **feature extraction**:  
Create new features from old ones.

👉 **PCA does exactly this**, but in a **systematic, mathematical** way for any number of dimensions.

---

## 4. What PCA Actually Does

Given a dataset with **d features**:

- PCA computes **d new axes** (directions) called **principal components (PCs)**.
- These are:
  - **Linear combinations** of original features.
  - **Orthogonal** to each other (uncorrelated).
  - Sorted by how much **variance** they capture.
```latex
If your original features are \( x_1, x_2, ..., x_d \),

each principal component \( \text{PC}_k \) is of the form:

\[
\text{PC}_k = a_{k1} x_1 + a_{k2} x_2 + \dots + a_{kd} x_d
\]

Where vector \( a_k = (a_{k1},...,a_{kd}) \) is a **direction** in feature space.

```
- **PC1**: direction with **maximum variance**.
- **PC2**: direction with **maximum variance** subject to being **orthogonal to PC1**.
- etc.

Then you:

- **Keep only the first K components** (`K < d`),
- Drop the remaining ones,
- Work with the transformed data in K dimensions.

So PCA:

- **Transforms** the coordinate system (rotates it),
- **Keeps the high-variance directions**,
- **Throws away low-variance directions** (assumed mostly noise/redundancy).

---

## 5. Geometric Intuition with Rooms/Bathrooms Example

Original 2D coordinates:

- x-axis: `rooms`
- y-axis: `bathrooms`

Points lie roughly along a diagonal:

```text
^ bathrooms
|
|      *
|    *   *
|  *       *
|*___________> rooms


```
PCA will:

- Rotate the axes so that:
  - **PC1** lies along the direction of **maximum spread** (the diagonal).
  - **PC2** is **perpendicular** to PC1.

So the new axes become:

- **PC1** ≈ “overall size / space” of the flat  
  (a linear combination of `rooms + bathrooms`)
- **PC2** ≈ small leftover variation  
  (noise or minor deviations from perfect correlation)

Now:

- **Variance along PC1** is very large.
- **Variance along PC2** is small.

So we can:

- **Keep only PC1**, **drop PC2**.

We reduced **2D → 1D**.

**Interpretation:**

- We converted two correlated features (`rooms`, `bathrooms`) into one main feature: a **size-like** direction.
- That’s **feature extraction via PCA** (not just dropping columns, but creating a new, more informative axis).


---

## 6. Why PCA Cares About Variance

You repeatedly saw statements like:

- “Pick axis with maximum spread / variance”
- “Maximize variance along principal components”

**Question:** Why is variance so important?

### 6.1 Mean vs Variance – Quick Recap

Given data points \( x_1, x_2, \dots, x_n \):

- **Mean:**

  \[
  \mu = \frac{1}{n} \sum_{i=1}^{n} x_i
  \]

- **Variance:**

  \[
  \operatorname{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2
  \]

- **Standard deviation:**

  \[
  \sigma = \sqrt{\operatorname{Var}(X)}
  \]

Two different datasets can have the **same mean** but very different **variance** (spread).

**Example:**

- Dataset A: \(-5,\; 0,\; +5\)
- Dataset B: \(-10,\; 0,\; +10\)

Both have mean \(= 0\), but:

- Dataset B is **more spread out** → **larger variance**.

**Takeaway:**

- Mean → tells you the **center**.
- Variance → tells you **how spread out** values are around the center.


### 6.2 Variance as “Information”

In PCA, we want to keep directions where:

- Points are **spread out**, **meaningfully different**.

We want to ignore directions where points are:

- Almost identical,
- Differences look like **noise** or **redundancy**.

**High variance direction:**

- Data points differ a lot → **more information**.

**Low variance direction:**

- Points are almost the same → likely **noise** or **redundant** information.

So PCA searches for directions that:

- **Maximize variance**  
  → preserve **structure, distances, separation** in the data.


### 6.3 Example with Two Classes (Green vs Red Points)

Imagine 2D data with two classes:

- Green points (class A)
- Red points (class B)

They are well separated **diagonally**, but depending on the projection axis:

- **Project onto a good direction**:
  - Green and red points remain **separated**.
  - Distances between classes are **preserved**.

- **Project onto a bad direction**:
  - Green and red points **collapse** close together.
  - A model finds it **harder** to separate them.

So, **maximizing variance** along the chosen direction helps:

- Maintain **distances and class separations**,  
- So that even after dimensionality reduction, the ML model still “sees” the **structure**.

That’s why PCA’s optimization goal is:

- **Find directions (vectors) that maximize the variance of projected data.**

Formally, for each direction \( w \):

\[
\max_{\|w\| = 1} \operatorname{Var}(X w)
\]

This leads to **eigenvectors of the covariance matrix**:

- These eigenvectors are the **principal components**.


---

## 7. Formal PCA Algorithm (Step-by-Step)

Assume your data matrix \( X \) has shape \((n_{\text{samples}}, n_{\text{features}})\).

### Step 1 – Standardize the Features

PCA is **scale-sensitive**, so:

1. Subtract the **mean** from each feature.
2. Optionally divide by the **standard deviation**.

Result: each feature has **mean 0**, and comparable scale.

### Step 2 – Compute the Covariance Matrix

For mean-centered data \( X \) of shape \( n \times d \):

\[
\Sigma = \frac{1}{n - 1} X^\top X
\]

- \( \Sigma \) is a \( d \times d \) **covariance matrix**.
- Entry \( \Sigma_{ij} \) is the covariance between feature \( i \) and \( j \).

### Step 3 – Eigen Decomposition

Compute **eigenvalues** and **eigenvectors** of \( \Sigma \):

\[
\Sigma v_k = \lambda_k v_k
\]

- \( v_k \): eigenvector (a **direction** in feature space).
- \( \lambda_k \): eigenvalue (variance **along that direction**).

### Step 4 – Sort by Eigenvalues

- Sort eigenvectors in **decreasing** order of their eigenvalues.
- Eigenvector with the **largest** eigenvalue → **PC1**.
- Next largest → **PC2**, and so on.

### Step 5 – Choose Number of Components \(K\)

Decide how many components to keep.

Common strategies:

- Keep enough components to explain, e.g., **95% of total variance**.
- Or choose \( K \) manually (e.g., \(K = 2\) or \(3\) for **visualization**).

### Step 6 – Project Original Data

Let \( W \) be the matrix of top \(K\) eigenvectors (shape \( d \times K \)).

Transform data:

\[
Z = X W
\]

- \( Z \) has shape \( n \times K \): data in the new **K-dimensional PCA space**.
- Each **column of \(Z\)** is one **principal component**.

---

## 8. Where PCA Fits in an ML Pipeline

Typical **supervised ML pipeline**:

1. **Train–test split** your data.
2. **Scale / standardize** your features.
3. **Fit PCA only on training features** \( X_{\text{train}} \):

   ```python
   pca.fit(X_train_scaled)

4. Transform both train and test using the fitted PCA:
    ```python
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

5. Train your model (e.g. logistic regression, SVM) on X_train_pca.

6. Evaluate on X_test_pca.

Important:

. Never fit PCA on the full dataset before splitting.

. That would leak information from test into train.

. Treat PCA as a preprocessing step that is learned only from training data.

---
## 9. Practical Example (Conceptual) – Handwritten Digits

1. Assume you have a dataset of handwritten digits (MNIST-like):

    Each image:
    28
    ×
    28
    28×28 pixels.

    . After flattening: 784 features per image.

    . This is high-dimensional.

2. Pipeline:

    Flatten images → vectors of length 784.

    Standardize the features.

3. Apply PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)  # or use explained variance to choose
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
```

    . Now each image is represented by 50 numbers instead of 784.

4. Train a classifier (logistic regression, SVM, etc.) on these 50D vectors.

5. Benefits:

    . Much faster training and prediction.

    . Often similar or better generalization (less overfitting).

    . You can visualize digits in 2D/3D by taking 2 or 3 principal components.

## 10. Summary – Key Takeaways

### Problem
- Too many features → **curse of dimensionality**
  - Worse model performance
  - Higher computation cost

---

### Dimensionality Reduction Approaches
1. **Feature Selection**
   - Drop some original features
   - No new features created

2. **Feature Extraction**
   - Create new features from existing ones
   - Information is redistributed, not discarded directly

---

### PCA (Principal Component Analysis)
- **Unsupervised** (does not use labels)
- A **feature extraction** technique
- **Linear** method
- Old, stable, and widely used in practice

---

### What PCA Does
- Finds new **orthogonal axes** called *principal components*
- Each component is a **linear combination** of original features
- Components are ordered by **variance**:
  - **PC1** → highest variance
  - **PC2** → second highest
  - and so on
- Keep only the first **K components**

---

### Intuition
- Like a photographer choosing the best angle to capture a **3D scene in a 2D photo**
- Or combining *rooms* + *bathrooms* into a single **“size”** feature

---

### Why Variance Matters
- High-variance directions:
  - Capture meaningful differences between data points
  - Preserve structure and potential class separation
- PCA explicitly selects directions that **maximize variance**

---

### Implementation Notes
- **Always standardize features** before applying PCA
- **Fit PCA on training data only**
- Use the fitted PCA model to:
  - Transform training data
  - Transform test data
- Choose number of components using:
  - Explained variance ratio
  - Domain requirements
