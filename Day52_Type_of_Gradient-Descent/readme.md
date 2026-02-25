# Gradient Descent — Types, Mathematics, and Multi-Variable Case
---

## 1. Introduction

Gradient Descent is an **optimization algorithm** used in many Machine Learning algorithms to reach an optimal solution.

---

## 2. What is Gradient Descent?

Gradient Descent is an **iterative optimization algorithm** used to minimize a loss function.

It is used in:
- Linear Regression
- Logistic Regression
- Neural Networks
- Deep Learning

Its goal is simple:  
**minimize the loss and reach the best possible parameters**.

---

## 3. Linear Regression Recap

For simple linear regression:

$$
y = mx + b
$$

Where:
- $m$ = slope  
- $b$ = intercept  

We do not know $m$ and $b$ initially.

So we:
1. Start with random values
2. Update them repeatedly
3. Reach optimal values step by step

---

## 4. Loss Function

We use Mean Squared Error (MSE):

$$
L(m,b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

This loss:
- Depends on parameters
- Is convex
- Has a single global minimum

---

## 5. Gradient Descent Update Rules

For slope $m$:

$$
m := m - \alpha \frac{\partial L}{\partial m}
$$

For intercept $b$:

$$
b := b - \alpha \frac{\partial L}{\partial b}
$$

Where:
- $\alpha$ is the learning rate

This equation **is Gradient Descent**.

---

## 6. Key Observation (Why Types Exist)

👉 **Mathematics remains the same**  
👉 **Only the amount of data used per update changes**

This single difference creates **three types of Gradient Descent**.

---

## 7. Types of Gradient Descent

1. **Batch Gradient Descent**
2. **Stochastic Gradient Descent**
3. **Mini-Batch Gradient Descent**

---

## 8. Batch Gradient Descent

### How It Works

- Uses **entire dataset**
- Updates parameters **once per iteration**

If dataset has 300 rows:
- Loss is computed using all 300 rows
- One update is performed

### Mathematical Form

$$
\frac{\partial L}{\partial m} = -2 \sum_{i=1}^{n} x_i (y_i - (mx_i + b))
$$

$$
\frac{\partial L}{\partial b} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))
$$

### Pros and Cons

- ✅ Stable
- ❌ Very slow
- ❌ High computation

Used only when dataset is **small**.

---

## 9. Stochastic Gradient Descent (SGD)

### How It Works

- Uses **one data point**
- Updates parameters after every row

If dataset has 300 rows:
- 300 updates per epoch

### Mathematical Form (Single Sample)

$$
\frac{\partial L}{\partial m} = -2 x_i (y_i - (mx_i + b))
$$

$$
\frac{\partial L}{\partial b} = -2 (y_i - (mx_i + b))
$$

### Pros and Cons

- ✅ Very fast
- ❌ Noisy
- ❌ Oscillates

Used for **very large datasets**.

---

## 10. Mini-Batch Gradient Descent

### Why It Exists

- Batch → slow  
- SGD → unstable  

Mini-batch balances both.

### How It Works

- Dataset is split into batches
- Batch size: 16, 32, 64, etc.

If dataset has 300 rows and batch size = 30:
- 10 updates per epoch

### Mathematical Form

$$
\frac{\partial L}{\partial m} = -2 \sum_{i \in batch} x_i (y_i - (mx_i + b))
$$

$$
\frac{\partial L}{\partial b} = -2 \sum_{i \in batch} (y_i - (mx_i + b))
$$

This is the **most used** method in practice.

---

## 11. Summary of Gradient Descent Types

| Type | Data per Update | Speed | Stability |
|----|----|----|----|
| Batch | Full dataset | Slow | High |
| SGD | One row | Fast | Low |
| Mini-Batch | Small batch | Fast | Balanced |

---

## 12. Multiple Linear Regression Case

When data has multiple features:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
$$

Number of parameters:

$$
\text{Total parameters} = n + 1
$$

---

## 13. Loss Function (Multi-Variable)

$$
L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

## 14. Gradient Vector

$$
\nabla L =
\begin{bmatrix}
\frac{\partial L}{\partial \beta_0} \\
\frac{\partial L}{\partial \beta_1} \\
\vdots \\
\frac{\partial L}{\partial \beta_n}
\end{bmatrix}
$$

Each parameter has its **own derivative**.

---

## 15. General Gradient Formula

For any parameter $$\beta_j$$:

$$
\frac{\partial L}{\partial \beta_j}
= -2 \sum_{i=1}^{n} (y_i - \hat{y}_i) x_{ij}
$$

---

## 16. Matrix Form (Efficient Implementation)

Prediction:

$$
\hat{y} = X\beta
$$

Gradient:

$$
\nabla L = -2 X^T (y - X\beta)
$$

This allows **vectorized computation**.

---

## 17. Final Takeaways

- Same math, different execution
- Mini-batch is most practical
- Learning rate controls convergence
- Gradient Descent scales to any number of features
- Foundation of Machine Learning optimization

---

