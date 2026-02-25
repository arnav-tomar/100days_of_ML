# Gradient Descent — Detailed Theory, Derivations, and Code

> **Scope rule followed:** Everything below is derived **only** from the material you provided. No external concepts, shortcuts, or extra theory have been added.

---

## 1. What Gradient Descent Is

Gradient Descent is a **first‑order optimization algorithm** used to find the **minimum of a differentiable function**.

Formally:

> Gradient Descent finds a (local) minimum of a function by taking **repeated steps in the opposite direction of the gradient** at the current point.

If the function is convex, this minimum is also the **global minimum**.

---

## 2. Why Gradient Descent Is Needed

In **linear regression**, parameters can be found using a **closed‑form solution** (normal equation). But:

* In **high dimensions**, matrix inversion becomes computationally expensive
* In **large datasets**, closed‑form solutions are slow
* In **logistic regression & deep learning**, no closed‑form solution exists

So we need an **iterative optimization method** → Gradient Descent.

---

## 3. Problem Setup (Linear Regression)

We use **linear regression** to build intuition.

### Model

$$
\hat{y} = mx + b
$$

Where:

* $m$ = slope
* $b$ = intercept

---

## 4. Loss (Cost) Function

To measure how bad a line is, we use **Mean Squared Error (MSE)** (without $\frac{1}{n}$ for simplicity, exactly as in your lecture):

$$
L(m,b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

This function:

* Depends on **parameters** $m$ and $b$
* Is **parabolic** (convex)
* Has **one minimum**

---

## 5. Key Idea of Gradient Descent

We want values of $m$ and $b$ such that:

$$
L(m,b) \rightarrow \min
$$

We:

1. Start with **random values** of $m$ and $b$
2. Compute the **slope of the loss function**
3. Move parameters in the **opposite direction of the slope**
4. Repeat

---

## 6. Slope and Derivatives

### What does slope tell us?

* **Positive slope** → decrease parameter
* **Negative slope** → increase parameter

Slope is obtained using **derivatives**.

---

## 7. Gradient Descent with One Parameter (Only $b$)

To simplify intuition, we **fix $m$** and optimize only $b$.

### Loss function (with fixed $m$):

$$
L(b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

This is a **parabola** in $b$.

---

## 8. Derivative w.r.t. $b$

$$
\frac{dL}{db} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))
$$

This derivative:

* Gives **direction**
* Gives **magnitude** of change

---

## 9. Update Rule (Core Formula)

$$
b_{new} = b_{old} - \alpha \frac{dL}{db}
$$

Where:

* $\alpha$ = **learning rate**

This equation **is Gradient Descent**.

---

## 10. Role of Learning Rate ($\alpha$)

### Too large

* Overshoots minimum
* Oscillation
* Divergence

### Too small

* Very slow convergence

### Proper value

* Fast and stable convergence

Step size:

$$
\text{Step size} = \alpha \times \text{slope}
$$

---

## 11. Stopping Criteria

Gradient Descent must stop when:

### Option 1: Parameter change is small

$$
|b_{new} - b_{old}| < \epsilon
$$

### Option 2: Fixed number of iterations

$$
\text{iterations} = K
$$

---

## 12. Full Gradient Descent (Both $m$ and $b$)

Now we optimize **both parameters**.

Loss function:

$$
L(m,b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

---

## 13. Partial Derivatives

### Derivative w.r.t. $b$

$$
\frac{\partial L}{\partial b} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))
$$

### Derivative w.r.t. $m$

$$
\frac{\partial L}{\partial m} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))x_i
$$

These two together form the **gradient vector**.

---

## 14. Update Rules

$$
b := b - \alpha \frac{\partial L}{\partial b}
$$

$$
m := m - \alpha \frac{\partial L}{\partial m}
$$

Both updates happen **simultaneously** per iteration.

---

## 15. Geometric Interpretation

* Loss function becomes a **3D surface**
* Shape is a **bowl (paraboloid)**
* Gradient is the **steepest ascent direction**
* Gradient descent moves **downhill**

---

## 16. Convex vs Non‑Convex Loss

### Convex loss (MSE)

* Single global minimum
* Guaranteed convergence

### Non‑convex loss

* Multiple local minima
* Gradient descent can get stuck

---

## 17. Effect of Feature Scaling

If features have very different scales:

* Contours become **elongated**
* Convergence becomes **slow and zig‑zag**

Solution:

* Normalize / standardize features before training

---

## 18. Variants Mentioned

* **Batch Gradient Descent** – uses full dataset
* **Stochastic Gradient Descent (SGD)** – one sample
* **Mini‑batch Gradient Descent** – subset of data

(Only names mentioned here, no extra theory added.)

---

## 19. Simple Python Code (From Your Explanation)

```python
import numpy as np

# sample data
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

m = 0.0
b = 0.0
alpha = 0.01
n = len(x)

for _ in range(1000):
    y_pred = m * x + b
    dm = -2 * np.sum((y - y_pred) * x)
    db = -2 * np.sum(y - y_pred)

    m = m - alpha * dm
    b = b - alpha * db

print(m, b)
```

---

## 20. Core Takeaways

* Gradient Descent is **iterative optimization**
* Moves parameters opposite to gradient
* Learning rate controls step size
* Works for **any differentiable loss function**
* Backbone of **machine learning & deep learning**

---

## 21. One‑Line Definition (Final)

> **Gradient Descent repeatedly updates parameters by subtracting the learning‑rate‑scaled gradient of the loss function to reach a minimum.**
# Gradient Descent — Detailed Theory, Derivations, and Code

> **Scope rule followed:** Everything below is derived **only** from the material you provided. No external concepts, shortcuts, or extra theory have been added.

---

## 1. What Gradient Descent Is

Gradient Descent is a **first‑order optimization algorithm** used to find the **minimum of a differentiable function**.

Formally:

> Gradient Descent finds a (local) minimum of a function by taking **repeated steps in the opposite direction of the gradient** at the current point.

If the function is convex, this minimum is also the **global minimum**.

---

## 2. Why Gradient Descent Is Needed

In **linear regression**, parameters can be found using a **closed‑form solution** (normal equation). But:

* In **high dimensions**, matrix inversion becomes computationally expensive
* In **large datasets**, closed‑form solutions are slow
* In **logistic regression & deep learning**, no closed‑form solution exists

So we need an **iterative optimization method** → Gradient Descent.

---

## 3. Problem Setup (Linear Regression)

We use **linear regression** to build intuition.

### Model

$$
\hat{y} = mx + b
$$

Where:

* $m$ = slope
* $b$ = intercept

---

## 4. Loss (Cost) Function

To measure how bad a line is, we use **Mean Squared Error (MSE)** (without $\frac{1}{n}$ for simplicity, exactly as in your lecture):

$$
L(m,b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

This function:

* Depends on **parameters** $m$ and $b$
* Is **parabolic** (convex)
* Has **one minimum**

---

## 5. Key Idea of Gradient Descent

We want values of $m$ and $b$ such that:

$$
L(m,b) \rightarrow \min
$$

We:

1. Start with **random values** of $m$ and $b$
2. Compute the **slope of the loss function**
3. Move parameters in the **opposite direction of the slope**
4. Repeat

---

## 6. Slope and Derivatives

### What does slope tell us?

* **Positive slope** → decrease parameter
* **Negative slope** → increase parameter

Slope is obtained using **derivatives**.

---

## 7. Gradient Descent with One Parameter (Only $b$)

To simplify intuition, we **fix $m$** and optimize only $b$.

### Loss function (with fixed $m$):

$$
L(b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

This is a **parabola** in $b$.

---

## 8. Derivative w.r.t. $b$

$$
\frac{dL}{db} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))
$$

This derivative:

* Gives **direction**
* Gives **magnitude** of change

---

## 9. Update Rule (Core Formula)

$$
b_{new} = b_{old} - \alpha \frac{dL}{db}
$$

Where:

* $\alpha$ = **learning rate**

This equation **is Gradient Descent**.

---

## 10. Role of Learning Rate ($\alpha$)

### Too large

* Overshoots minimum
* Oscillation
* Divergence

### Too small

* Very slow convergence

### Proper value

* Fast and stable convergence

Step size:

$$
\text{Step size} = \alpha \times \text{slope}
$$

---

## 11. Stopping Criteria

Gradient Descent must stop when:

### Option 1: Parameter change is small

$$
|b_{new} - b_{old}| < \epsilon
$$

### Option 2: Fixed number of iterations

$$
\text{iterations} = K
$$

---

## 12. Full Gradient Descent (Both $m$ and $b$)

Now we optimize **both parameters**.

Loss function:

$$
L(m,b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

---

## 13. Partial Derivatives

### Derivative w.r.t. $b$

$$
\frac{\partial L}{\partial b} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))
$$

### Derivative w.r.t. $m$

$$
\frac{\partial L}{\partial m} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))x_i
$$

These two together form the **gradient vector**.

---

## 14. Update Rules

$$
b := b - \alpha \frac{\partial L}{\partial b}
$$

$$
m := m - \alpha \frac{\partial L}{\partial m}
$$

Both updates happen **simultaneously** per iteration.

---

## 15. Geometric Interpretation

* Loss function becomes a **3D surface**
* Shape is a **bowl (paraboloid)**
* Gradient is the **steepest ascent direction**
* Gradient descent moves **downhill**

---

## 16. Convex vs Non‑Convex Loss

### Convex loss (MSE)

* Single global minimum
* Guaranteed convergence

### Non‑convex loss

* Multiple local minima
* Gradient descent can get stuck

---

## 17. Effect of Feature Scaling

If features have very different scales:

* Contours become **elongated**
* Convergence becomes **slow and zig‑zag**

Solution:

* Normalize / standardize features before training

---

## 18. Variants Mentioned

* **Batch Gradient Descent** – uses full dataset
* **Stochastic Gradient Descent (SGD)** – one sample
* **Mini‑batch Gradient Descent** – subset of data

(Only names mentioned here, no extra theory added.)

---

## 19. Simple Python Code (From Your Explanation)

```python
import numpy as np

# sample data
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

m = 0.0
b = 0.0
alpha = 0.01
n = len(x)

for _ in range(1000):
    y_pred = m * x + b
    dm = -2 * np.sum((y - y_pred) * x)
    db = -2 * np.sum(y - y_pred)

    m = m - alpha * dm
    b = b - alpha * db

print(m, b)
```

---

---

## 20. Learning Rate (α), Step Size, and Why It Is Needed

Up to now, parameters were updated directly using the derivative.  
This causes **very large jumps**, which can overshoot the minimum.

To control how big each update is, we introduce the **learning rate** (also called step size), denoted by **α**.

### Core Update Rule

$$
b_{new} = b_{old} - \alpha \frac{dL}{db}
$$

Where:
- α is a small positive number
- Typical values: `0.1`, `0.01`, `0.001`

### Effect of Learning Rate

- **α too large** → oscillation or divergence  
- **α too small** → very slow convergence  

---

## 21. Numerical Example (Single Parameter: b)

Assume:
- Initial value: \( b = -10 \)
- Slope: \( \frac{dL}{db} = -5 \)
- Learning rate: \( \alpha = 0.1 \)

Update:

$$
b_{new} = -10 - 0.1(-5) = -9.5
$$

Repeating this process moves \( b \) step-by-step toward the minimum.

---

## 22. Stopping Criteria

Gradient Descent must stop using a condition.

### Method 1: Small Parameter Change

$$
|b_{new} - b_{old}| < \epsilon
$$

### Method 2: Fixed Number of Iterations

Run the algorithm for a fixed number of iterations:

- iterations = 100 / 500 / 1000


---

## 23. Gradient Descent Algorithm (Only b)

Steps:
1. Initialize \( b \)
2. Compute derivative \( \frac{dL}{db} \)
3. Update \( b \) using learning rate
4. Repeat until stopping condition

Mathematical form:

$$
b := b - \alpha \left(-2 \sum_{i=1}^{n} (y_i - (mx_i + b))\right)
$$

---

## 24. Python Code: Gradient Descent for Only b

```python
import numpy as np

X = np.array([1, 2, 3, 4])
Y = np.array([2, 4, 6, 8])

m = 2.0
b = -10.0
alpha = 0.1

for i in range(20):
    y_pred = m * X + b
    db = -2 * np.sum(Y - y_pred)
    b = b - alpha * db
    print(i, b)
```

---

---

## 25. Gradient Descent with Two Parameters (m and b)

Until now, we optimized only one parameter.  
Now **both slope (m) and intercept (b) are unknown** and must be optimized together.

### Loss Function

$$
L(m,b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$

This loss depends on **two parameters**, so the loss surface becomes **2D in parameters and 3D geometrically**.

### Partial Derivatives

Derivative with respect to **b**:

$$
\frac{\partial L}{\partial b} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))
$$

Derivative with respect to **m**:

$$
\frac{\partial L}{\partial m} = -2 \sum_{i=1}^{n} x_i (y_i - (mx_i + b))
$$

---

## 26. Gradient Vector (Why It Is Called Gradient)

When a function depends on **multiple parameters**, derivatives are combined into a **vector** called the **gradient**.

$$
\nabla L =
\begin{bmatrix}
\frac{\partial L}{\partial m} \\
\frac{\partial L}{\partial b}
\end{bmatrix}
$$

Properties:
- Gradient points in the direction of **maximum increase** of loss
- Magnitude tells how steep the increase is
- Gradient Descent moves in the **opposite direction** of this vector

That is why it is called **Gradient Descent**.

---

## 27. Update Rules (Two Parameters)

Both parameters are updated **simultaneously** in each iteration.

Update rule for **m**:

$$
m := m - \alpha \frac{\partial L}{\partial m}
$$

Update rule for **b**:

$$
b := b - \alpha \frac{\partial L}{\partial b}
$$

Where:
- α is the learning rate
- Updates use the **current values** of m and b
- Repeating these updates moves parameters toward the minimum

---

28. Python Code: Gradient Descent for m and b
m = 0.0
b = 0.0
alpha = 0.01

for i in range(100):
    y_pred = m * X + b
    dm = -2 * np.sum(X * (Y - y_pred))
    db = -2 * np.sum(Y - y_pred)

    m = m - alpha * dm
    b = b - alpha * db

print(m, b)

---

29. Geometric Interpretation

Loss surface is a bowl (paraboloid)

Far from minimum → large steps

Near minimum → small steps
