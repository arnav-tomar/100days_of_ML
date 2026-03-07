<div align="center">

# 🧠 Machine Learning — Handwritten Notes Repository

[![Author](https://img.shields.io/badge/Author-Arnav%20Tomar%2018-blue?style=for-the-badge&logo=github)](https://github.com/)
[![Type](https://img.shields.io/badge/Type-Academic%20Study%20Guide-green?style=for-the-badge&logo=bookstack)](https://github.com/)
[![Topics](https://img.shields.io/badge/Topics-60%2B%20Concepts-orange?style=for-the-badge&logo=academia)](https://github.com/)
[![Math](https://img.shields.io/badge/Contains-Math%20Derivations-red?style=for-the-badge&logo=overleaf)](https://github.com/)

> *A structured, page-by-page index of handwritten ML notes — covering everything from linear algebra foundations to ensemble methods and kernel machines.*

---

</div>

## 📋 Table of Contents

- [Module 1 — Foundations of Data Science & AI](#-module-1--foundations-of-data-science--ai)
- [Module 2 — Model Architecture & Evaluation](#-module-2--model-architecture--evaluation)
- [Module 3 — Dimensionality Reduction (PCA)](#-module-3--dimensionality-reduction-pca)
- [Module 4 — Regression Analysis](#-module-4--regression-analysis-math-derivations)
- [Module 5 — Regularization & Classification](#-module-5--regularization--classification)
- [Module 6 — Tree-Based & Ensemble Methods](#-module-6--tree-based--ensemble-methods)
- [Module 7 — Unsupervised Learning & SVM](#-module-7--unsupervised-learning--svm)

---

## 📂 Module 1 — Foundations of Data Science & AI

> *Establishes the landscape of AI, ML, and Deep Learning with a clear taxonomy of learning paradigms and deployment strategies.*

---

### 📄 Page 1 — Data Science Overview

**What's covered:** Defines Data Science as a discipline and maps out the core toolkit. Lists essential libraries — **Pandas**, **NumPy**, and **Scikit-learn** — alongside the overall study roadmap for the notes. Acts as a north star for the entire document.

---

### 📄 Page 2 — AI vs. ML vs. Deep Learning

**What's covered:** Builds a hierarchical definition: AI ⊃ ML ⊃ DL. Introduces the two primary learning types at a high level — **Supervised Learning** (labeled data, predict outcomes) and **Unsupervised Learning** (unlabeled data, find structure). Sets the conceptual frame for all algorithms that follow.

---

### 📄 Page 3 — Learning Paradigms

**What's covered:** Categorizes all ML algorithms into four fundamental paradigms:
- **Classification** — Predict a discrete class label
- **Regression** — Predict a continuous numeric value
- **Clustering** — Group similar data points without labels
- **Association** — Discover rules linking variables (e.g., market basket)

---

### 📄 Page 4 — Deep Learning Intro

**What's covered:** Introduces the **Perceptron** as the fundamental building block of neural networks. Includes a summary reference table for edge-case paradigms: **Anomaly Detection** and **Rule-Based Learning**, distinguishing them from the core four.

---

### 📄 Pages 5–6 — ML Deployment Types

**What's covered:** A deep dive into two contrasting production strategies:
- **Batch / Offline ML** — Model trained on static historical data, predictions run in bulk
- **Online Machine Learning** — Model updates incrementally as new data arrives

Covers retraining schedules, data drift considerations, and when to choose each strategy for real-world deployment.

---

## 📂 Module 2 — Model Architecture & Evaluation

> *Compares foundational model types, dissects common data quality pitfalls, and maps the full ML development lifecycle.*

---

### 📄 Pages 7–8 — Instance-Based vs. Model-Based Learning

**What's covered:** A head-to-head comparison of two architectural philosophies:
- **Instance-Based (KNN)** — Memorizes training data; prediction via similarity at inference time
- **Model-Based (Linear Regression, Decision Trees)** — Learns a compact mathematical model from data

Also covers structured vs. unstructured data collection methods.

---

### 📄 Pages 9–10 — Data Challenges

**What's covered:** Analyzes the most dangerous failure modes in ML projects:
- **Sampling Bias** — Training distribution doesn't match real-world distribution
- **Overfitting** — Model memorizes noise; high variance, low bias
- **Underfitting** — Model too simple; high bias, low variance

Grounds these concepts in industrial applications across **Retail** and **Banking** sectors.

---

### 📄 Pages 11–12 — MLDLC / MLOps

**What's covered:** Step-by-step walk through the full **Machine Learning Development Lifecycle**:

`Problem Framing` → `Data Gathering` → `Preprocessing` → `EDA` → `Modeling` → `Evaluation` → `A/B Testing` → `Deployment` → `Monitoring`

Emphasizes the iterative, non-linear nature of real ML pipelines and introduces MLOps principles.

---

### 📄 Pages 13–16 — Tensor Mathematics

**What's covered:** Rigorous definitions of tensor structures as the mathematical backbone of ML:

| Tensor | Rank | Shape Example | Use Case |
|--------|------|---------------|----------|
| **Scalar** | 0D | `()` | Single loss value |
| **Vector** | 1D | `(n,)` | Feature row |
| **Matrix** | 2D | `(m, n)` | Dataset, weight layers |
| **N-Tensor** | ND | `(b, h, w, c)` | Image batches, video |

Covers rank, shape, axes, and vectorization strategies for **NLP** (token embeddings) and **Time-Series** (sliding windows).

---

## 📂 Module 3 — Dimensionality Reduction (PCA)

> *A mathematically rigorous treatment of Principal Component Analysis — from the curse of dimensionality through eigendecomposition.*

---

### 📄 Page 19 — Curse of Dimensionality

**What's covered:** Explains why high-dimensional spaces become problematic — data grows exponentially **sparse**, distance metrics lose meaning, and models overfit. Motivates the need for feature extraction techniques like **PCA** and **LDA** over naive feature selection.

---

### 📄 Pages 20–22 — PCA Core Mechanics

**What's covered:** The geometric and algebraic foundations of PCA:
- **Vector Projection** — Projecting data points onto candidate axes
- **Unit Vectors** — Constraining principal component directions to length 1
- **Variance Maximization** — The core objective: find the axis along which data spreads the most

$$\text{maximize} \quad \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \|\mathbf{w}\| = 1$$

---

### 📄 Pages 23–25 — Eigendecomposition

**What's covered:** The mathematical engine behind PCA:
- **Covariance Matrix** construction: $\Sigma = \frac{1}{n} X^T X$
- **Linear Transformations** and what they do to vector space
- **Eigenvalues** ($\lambda$) — How much variance each component captures
- **Eigenvectors** — The directions (principal components) themselves

$$\Sigma \mathbf{v} = \lambda \mathbf{v}$$

Differentiates between covariance and correlation matrices and explains when to standardize before PCA.

---

### 📄 Page 26 — PCA Implementation

**What's covered:** Translates the math into practical Python code using `np.linalg.eig`. Covers the full pipeline: center data → compute covariance → extract eigenpairs → sort by eigenvalue → project data onto top-k components.

```python
# Core PCA step
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
```

---

## 📂 Module 4 — Regression Analysis (Math Derivations)

> *The most math-heavy module — derives regression from first principles using calculus, matrix algebra, and iterative optimization.*

---

### 📄 Pages 27–30 — Simple Linear Regression

**What's covered:** Derives the fundamental regression equation $y = mx + b$ from scratch. Visualizes the **Loss Function (MSE)** as a convex bowl in parameter space, building intuition for why optimization converges to a unique global minimum.

$$\mathcal{L}(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

---

### 📄 Pages 31–33 — Ordinary Least Squares (OLS) Proof

**What's covered:** A rigorous calculus-based proof — takes partial derivatives of the MSE loss with respect to both $m$ and $b$, sets them to zero, and derives the closed-form **OLS solution**:

$$m = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}, \qquad b = \bar{y} - m\bar{x}$$

---

### 📄 Pages 34–37 — Regression Metrics

**What's covered:** Deep dive into five essential evaluation metrics with formulas and intuitions:

| Metric | Formula | Sensitivity |
|--------|---------|-------------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust to outliers |
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as target |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained |
| **Adjusted R²** | Penalizes extra features | Controls model complexity |

---

### 📄 Pages 38–43 — Multiple Linear Regression & Matrix Form

**What's covered:** Extends regression to $n$ features using matrix algebra. Derives the **Normal Equation** — the closed-form matrix solution that bypasses iterative optimization:

$$\boldsymbol{\beta} = (X^T X)^{-1} X^T Y$$

Covers matrix differentiation, the design matrix $X$, and when the normal equation fails (singular $X^T X$).

---

### 📄 Pages 44–48 — Gradient Descent

**What's covered:** The iterative optimization algorithm that powers modern ML. Derives the parameter update rule and analyzes the role of **Learning Rate ($\eta$)**:

$$\theta := \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta)$$

- Too large $\eta$ → overshoots, diverges
- Too small $\eta$ → converges extremely slowly
- Just right → smooth convergence to minimum

Formulates **Batch Gradient Descent** (uses entire dataset per update step).

---

### 📄 Pages 52–54 — Stochastic & Mini-Batch GD

**What's covered:** Solves Batch GD's computational bottleneck on large datasets:
- **SGD (Stochastic GD)** — One sample per update; noisy but fast
- **Mini-Batch GD** — $k$ samples per update; balances speed and stability

Covers **learning rate schedules** (step decay, exponential decay) and momentum-based variants.

---

### 📄 Page 55 — Polynomial Regression

**What's covered:** Extends linear regression to capture **non-linear relationships** by engineering higher-degree features. A degree-2 polynomial fit:

$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2$$

Discusses the risk of **overfitting** as degree increases and the role of regularization (covered next module) to control complexity.

---

## 📂 Module 5 — Regularization & Classification

> *Introduces penalty terms to control model complexity, then builds the full mathematical framework for binary and multi-class classification.*

---

### 📄 Pages 57–64 — Ridge Regression (L2 Regularization)

**What's covered:** Adds an L2 penalty to the loss function to **shrink** regression coefficients and reduce variance:

$$\mathcal{L}_{Ridge} = \text{MSE} + \lambda \sum_{j=1}^{p} \beta_j^2$$

Derives the modified normal equation: $\boldsymbol{\beta} = (X^T X + \lambda I)^{-1} X^T Y$. Deep treatment of the **Bias-Variance Tradeoff** — how increasing $\lambda$ trades lower variance for higher bias.

---

### 📄 Pages 66–68 — Lasso Regression (L1 Regularization)

**What's covered:** Replaces the L2 penalty with an L1 (absolute value) penalty, which induces **sparsity** — driving some coefficients exactly to zero:

$$\mathcal{L}_{Lasso} = \text{MSE} + \lambda \sum_{j=1}^{p} |\beta_j|$$

This property makes Lasso an automatic **feature selection** tool — irrelevant features are eliminated from the model entirely.

---

### 📄 Page 69 — Elastic Net

**What's covered:** Combines L1 and L2 penalties into a single regularizer, controlled by a mixing ratio $\rho$:

$$\mathcal{L}_{EN} = \text{MSE} + \lambda \left[ \rho \sum|\beta_j| + (1-\rho) \sum\beta_j^2 \right]$$

Particularly effective for **high multicollinearity** datasets where Lasso's arbitrary feature selection among correlated features is undesirable.

---

### 📄 Pages 70–75 — Logistic Regression

**What's covered:** Derives the **Sigmoid function** and its derivative — the core of binary classification:

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \sigma'(x) = \sigma(x)(1 - \sigma(x))$$

Covers the **Perceptron trick** as a precursor to gradient-based updates, decision boundaries, and probability calibration.

---

### 📄 Pages 76–81 — Binary Cross-Entropy Loss

**What's covered:** Derives the **Log-Loss** (Binary Cross-Entropy) as the proper loss function for probabilistic classification:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

Extends gradient descent to this loss in full matrix form, deriving update rules for the weight vector $\mathbf{w}$.

---

### 📄 Pages 82–87 — Classification Metrics

**What's covered:** A comprehensive framework for evaluating classifiers beyond raw accuracy:

| Metric | Formula | When to prioritize |
|--------|---------|-------------------|
| **Accuracy** | $(TP+TN)$ / Total | Balanced classes |
| **Precision** | $TP / (TP+FP)$ | Cost of false positives is high |
| **Recall** | $TP / (TP+FN)$ | Cost of false negatives is high |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P+R}$ | Imbalanced classes |

Covers **Confusion Matrix** construction and **Macro/Micro/Weighted** averaging for multi-class scenarios.

---

### 📄 Pages 88–91 — Softmax Regression

**What's covered:** Generalizes logistic regression to $K$ classes using the **Softmax function**:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

Introduces **One-Hot Encoding (OHE)** for representing categorical targets and derives **Categorical Cross-Entropy** as the corresponding multi-class loss.

---

## 📂 Module 6 — Tree-Based & Ensemble Methods

> *From a single decision tree to the most powerful ensemble algorithms — AdaBoost, Gradient Boosting, and XGBoost.*

---

### 📄 Pages 94–98 — Decision Trees

**What's covered:** The mathematics of tree construction — how to choose the best feature to split on at each node:

$$\text{Entropy: } E(S) = -\sum_{i} p_i \log_2 p_i$$
$$\text{Information Gain: } IG = E(\text{parent}) - \sum_{\text{children}} w_k \cdot E(k)$$

Covers **Gini Impurity** as an alternative criterion and extends the framework to **Regression Trees** (predict mean of leaf node).

---

### 📄 Pages 99–101 — Bagging (Bootstrap Aggregation)

**What's covered:** Reduces variance by training multiple models on **bootstrapped** (random sample with replacement) subsets of data and averaging their predictions. Establishes the conceptual groundwork for **Random Forest** — bagging applied specifically to decision trees.

---

### 📄 Pages 102–103 — Random Forest

**What's covered:** Adds **random feature subsampling** at each split to decorrelate individual trees. Covers key hyperparameters (`n_estimators`, `max_depth`, `max_features`) and the mathematical formula for **Feature Importance** based on mean decrease in impurity.

---

### 📄 Pages 104–108 — AdaBoost

**What's covered:** Sequential **boosting** — each learner focuses on what the previous one got wrong:

1. Initialize equal sample weights: $w_i = \frac{1}{n}$
2. Train a **weak learner** (Decision Stump — depth-1 tree)
3. Compute learner weight: $\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}$
4. **Upweight** misclassified samples, downweight correct ones
5. Repeat; final prediction = weighted vote

---

### 📄 Pages 110–118 — Gradient Boosting

**What's covered:** The most mathematically rich section of Module 6. Builds an additive ensemble stage-by-stage by fitting each new tree to the **pseudo-residuals** (negative gradient of the loss):

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $h_m$ is trained on $-\frac{\partial \mathcal{L}}{\partial F_{m-1}}$. Covers calculus-based derivation, learning rate shrinkage, and subsampling for regularization.

---

### 📄 Pages 122–124 — Stacking & Blending

**What's covered:** A meta-ensemble strategy that trains a **meta-model** (Level-1 learner) on the out-of-fold predictions of base learners (Level-0). Differentiates between:
- **Stacking** — Uses k-fold CV to generate meta-features
- **Blending** — Uses a held-out validation set instead

---

### 📄 Page 144 — XGBoost

**What's covered:** Production-grade gradient boosting with system-level optimizations:
- **Parallel tree construction** using column-block data structure
- **Regularized objective** with both L1 and L2 penalties on leaf weights
- **Tree pruning** with `max_depth` and `gamma` (minimum gain to make a split)
- **Sparsity-aware split finding** for missing data

---

## 📂 Module 7 — Unsupervised Learning & SVM

> *Covers the full spectrum of unsupervised methods alongside the elegant geometric framework of Support Vector Machines.*

---

### 📄 Page 109 — K-Means Clustering

**What's covered:** The canonical unsupervised algorithm:
1. Initialize $k$ centroids randomly (or via **K-Means++** for smarter initialization)
2. Assign each point to the nearest centroid (Euclidean distance)
3. Recompute centroids as the mean of assigned points
4. Repeat until convergence

$$\text{Objective: minimize} \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$

---

### 📄 Pages 125–127 — Hierarchical Clustering

**What's covered:** Builds a tree of clusters (**Dendrogram**) without requiring $k$ to be specified upfront:
- **Agglomerative (Bottom-Up)** — Start with each point as its own cluster; merge iteratively
- **Divisive (Top-Down)** — Start with one cluster; split iteratively

Covers linkage criteria: **Single**, **Complete**, **Average**, and **Ward's** method.

---

### 📄 Pages 128–130 — KNN Deep Dive

**What's covered:** Fully unpacks **K-Nearest Neighbors** as a non-parametric, **lazy learner**:
- **Euclidean distance** as the default metric: $d = \sqrt{\sum_i (x_i - z_i)^2}$
- Choosing optimal $K$: Heuristic ($K = \sqrt{n}$) vs. cross-validation grid search
- **Curse of dimensionality** revisited — why KNN degrades in high dimensions

---

### 📄 Pages 131–132 — Linear Regression Assumptions

**What's covered:** Diagnostic checks every regression model should pass:
- **Homoscedasticity** — Residual variance is constant (check: Residuals vs. Fitted plot)
- **No Autocorrelation** — Residuals are independent (check: Durbin-Watson test)
- **No Multicollinearity** — Features are not highly correlated with each other (check: VIF score)
- **Normality of Residuals** (check: Q-Q plot)

---

### 📄 Pages 133–138 — Support Vector Machines (SVM)

**What's covered:** The geometric theory of maximum-margin classification:

$$\text{Maximize margin: } \frac{2}{\|\mathbf{w}\|} \quad \text{subject to: } y_i(\mathbf{w} \cdot x_i + b) \geq 1$$

- **Hard Margin SVM** — Perfectly linearly separable data; no misclassifications allowed
- **Soft Margin SVM** — Introduces slack variables $\xi_i$ and regularization parameter $C$ to allow some violations
- **Hinge Loss**: $\mathcal{L} = \max(0, 1 - y_i \hat{y}_i)$

---

### 📄 Page 139 — The Kernel Trick

**What's covered:** Enables SVM to classify **non-linearly separable** data by implicitly mapping features to a higher-dimensional space — without ever computing the transformation explicitly:

| Kernel | Formula | Best for |
|--------|---------|---------|
| **Linear** | $\mathbf{x}^T \mathbf{z}$ | Linearly separable data |
| **Polynomial** | $(\gamma \mathbf{x}^T \mathbf{z} + r)^d$ | Moderate non-linearity |
| **RBF / Gaussian** | $\exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ | General non-linear boundaries |
| **Sigmoid** | $\tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ | Neural-network-like boundaries |

---

### 📄 Pages 140–143 — Naive Bayes

**What's covered:** A probabilistic classifier grounded in **Bayes' Theorem**:

$$P(C \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}$$

The "naive" assumption — **conditional independence** of features given the class — makes the likelihood tractable:

$$P(\mathbf{x} \mid C) = \prod_{i=1}^{n} P(x_i \mid C)$$

Covers three variants: **Gaussian NB** (continuous features, assumes normal distribution), **Multinomial NB** (word counts), and **Bernoulli NB** (binary features).

---

<div align="center">

## 📊 Coverage Summary

| Module | Pages | Core Concepts |
|--------|-------|---------------|
| 1 — Foundations | 1–6 | AI/ML taxonomy, deployment types |
| 2 — Architecture & Evaluation | 7–16 | Model types, MLDLC, Tensors |
| 3 — Dimensionality Reduction | 19–26 | PCA, Eigendecomposition |
| 4 — Regression | 27–55 | OLS, GD, SGD, Polynomial |
| 5 — Regularization & Classification | 57–91 | Ridge, Lasso, Logistic, Softmax |
| 6 — Tree & Ensemble Methods | 94–144 | Trees, RF, AdaBoost, GBM, XGBoost |
| 7 — Unsupervised & SVM | 109–143 | KMeans, KNN, SVM, Kernels, NB |

---

*Maintained by **Arnav Tomar 18** · Academic Study Reference · Not for redistribution*

</div>
