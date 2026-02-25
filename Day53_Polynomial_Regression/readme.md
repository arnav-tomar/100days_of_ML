# Polynomial Regression — Theory, Intuition, Degree Selection, and Overfitting

---

## 1. Why Linear Regression Fails on Non-Linear Data

Till now, we assumed that the relationship between input and output is linear.

That assumption means:

$$
y = \beta_0 + \beta_1 x
$$

This works **only if the true relationship is a straight line**.

---

### Problem Statement

Suppose the true data-generating process is:

$$
y = 2.8x^2 + 0.8x + 2 + \text{noise}
$$

This is **clearly non-linear**.

If we apply simple linear regression here, the model will try to fit a **straight line** on a **curved pattern**, which is mathematically incorrect.

---

## 2. Visual Failure of Linear Regression

When linear regression is applied to non-linear data:

- Predictions are systematically wrong
- Error remains high
- Line cannot bend to match curvature

Reason:
$$
\text{A linear model cannot represent } x^2 \text{ behavior}
$$

---

## 3. Core Idea of Polynomial Regression

Polynomial regression **does NOT change the algorithm**.

It changes only the **features**.

We convert input features into **polynomial features**, then apply **linear regression**.

---

## 4. Polynomial Regression Equation

For degree-2 polynomial regression:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2
$$

For degree-3:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3
$$

General form:

$$
y = \beta_0 + \sum_{k=1}^{d} \beta_k x^k
$$

---

## 5. Important Clarification (Very Important)

Polynomial regression is **still linear regression**, because:

- Model is linear in **parameters**
- Non-linearity exists only in **features**

Example:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2
$$

This is linear in:
$$
\beta_0,\ \beta_1,\ \beta_2
$$

---

## 6. Feature Transformation Concept

Original input:

$$
X = [x]
$$

Polynomial transformation (degree = 2):

$$
X' = [1,\ x,\ x^2]
$$

Polynomial transformation (degree = 3):

$$
X' = [1,\ x,\ x^2,\ x^3]
$$

---

## 7. Why Target Variable Is NOT Transformed

We apply polynomial transformation **only on inputs**, not outputs.

Correct:
$$
x \rightarrow [x,\ x^2,\ x^3]
$$

Wrong:
$$
y \rightarrow y^2
$$

Because regression minimizes error on original target scale.

---

## 8. Fitting Polynomial Regression

After transformation, we apply standard linear regression:

$$
\hat{y} = X'\beta
$$

Cost function remains same:

$$
J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

## 9. Effect of Polynomial Degree

### Degree = 1
$$
\text{Underfitting}
$$

Model too simple → cannot capture curvature.

---

### Degree = 2
$$
\text{Good fit (if true relation is quadratic)}
$$

Captures curvature properly.

---

### Degree = 3 or 4
$$
\text{May still work, but risk increases}
$$

---

### Degree Very Large (e.g., 15, 30, 300)

$$
\textbf{Overfitting}
$$

Model memorizes training data noise.

---

## 10. Underfitting vs Overfitting

### Underfitting
- Degree too low
- Misses important patterns
- High bias

---

### Overfitting
- Degree too high
- Fits noise
- Poor test performance
- High variance

---

## 11. Overfitting Visualization Logic

Training data (blue points):
- Model fits extremely well

Test data (green points):
- Predictions become unstable

Reason:
$$
\text{Model learns noise, not true pattern}
$$

---

## 12. Why Higher Degree Causes Overfitting

Number of features grows rapidly.

For one input feature:

$$
\text{Degree } d \Rightarrow d \text{ features}
$$

For two input features:

$$
\text{Degree } 2 \Rightarrow 6 \text{ features}
$$

---

## 13. Polynomial Regression with Multiple Inputs

Let inputs be:

$$
x_1,\ x_2
$$

Degree-2 polynomial features:

$$
[1,\ x_1,\ x_2,\ x_1^2,\ x_2^2,\ x_1x_2]
$$

Model becomes:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2
+ \beta_3x_1^2 + \beta_4x_2^2 + \beta_5x_1x_2
$$

---

## 14. Degree Definition in Multivariate Case

Polynomial degree is:

$$
\text{Maximum sum of powers in any term}
$$

Example:

$$
x_1^2 x_2 \Rightarrow \text{degree } 3
$$

---

## 15. Why Degree Does NOT Increase Arbitrarily

Higher degree means:
- More parameters
- More variance
- Higher chance of overfitting

Hence:

$$
\text{Higher degree} \neq \text{better model}
$$

---

## 16. Choosing the Right Degree

Correct degree depends on:
- Data complexity
- Noise level
- Number of samples

---

### Practical Techniques

1. Train–test error comparison
2. Cross-validation
3. Validation curves
4. Bias-variance analysis

---

## 17. Key Insight from the Lecture

For the shown dataset:

$$
\textbf{Optimal degree} = 2
$$

Because:
- Data generated using quadratic equation
- Degree-2 captures true structure
- Higher degrees only add noise fitting

---

## 18. Final Understanding

- Polynomial regression solves **non-linear relationships**
- It is linear regression on **expanded features**
- Degree selection is critical
- Too low → underfitting
- Too high → overfitting

---

## 19. What You Must Practice

1. Try different degrees
2. Observe training vs test error
3. Visualize curves
4. Understand bias–variance tradeoff

---

## 20. Closing Note

Polynomial regression is powerful,
but **dangerous if misused**.

Always remember:

$$
\text{Simple model that generalizes > complex model that memorizes}
$$

---

