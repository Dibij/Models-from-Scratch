## Logistic Regression — In-Depth Notes (Mathematics + Code Formulas)

---

## Main Concepts

Logistic Regression is a **classification algorithm** that predicts the probability that an input belongs to a certain class.  
It does not directly output a class label like 0 or 1; instead, it outputs a **probability between 0 and 1**.  
The classification decision is then made by applying a threshold, typically 0.5.

Although the model looks similar to linear regression in structure, it is different because the output is passed through a **Sigmoid function**, ensuring it stays between 0 and 1.

---

## Workflow Steps

---

### Step 1: Construct a Linear Equation

First, combine all input features into a weighted sum, including the bias term.

```bash
z = b0 + b1 * x1 + b2 * x2 + ... + bn * xn
```
where:  
- `b0` is the intercept (bias)
- `b1, b2, ..., bn` are the learned weights
- `x1, x2, ..., xn` are the input features

---

### Step 2: Apply the Sigmoid Activation

To map the raw score (`z`) into a probability range, apply the Sigmoid function.

```bash
sigmoid(z) = 1 / (1 + exp(-z))
```
The exponential function `exp(-z)` ensures that very large positive or negative values are smoothly mapped between 0 and 1.

---

### Step 3: Generate Final Prediction

The final predicted output `y_pred` becomes:

```bash
y_pred = sigmoid(z)
```

The interpretation is:  
- If `y_pred > 0.5`, the model predicts class **1**  
- If `y_pred <= 0.5`, the model predicts class **0**

Thresholds can be changed depending on the business use case, but 0.5 is the default.

---

## Mathematical Details

---

### Binary Cross-Entropy Loss (Log Loss)

Logistic regression models are trained to minimize the **Binary Cross-Entropy Loss**, also known as Log Loss.  
This loss measures the distance between the predicted probabilities and the true labels.

For a single training example:

```bash
Loss = -( y * log(y_pred) + (1 - y) * log(1 - y_pred) )
```
where:  
- `y` is the actual label (0 or 1)  
- `y_pred` is the predicted probability (a number between 0 and 1)

---

### Cost Function for the Entire Dataset

To compute the loss across all training examples, take the average:

```bash
J(θ) = -(1/m) * Σ [ y * log(y_pred) + (1 - y) * log(1 - y_pred) ]
```
where:  
- `m` is the total number of training samples  
- `Σ` denotes the summation over all training examples

This cost function needs to be minimized during training.

---

## Optimization Procedure

To minimize the cost function, we use **Gradient Descent**.  
The goal is to update the weights and bias so that the loss becomes smaller after every iteration.

---

### Gradient Calculation

The gradients (partial derivatives) for each parameter are computed as:

```bash
∂Loss/∂w = (1/m) * Σ (y_pred - y) * x
∂Loss/∂b = (1/m) * Σ (y_pred - y)
```

These gradients indicate how the cost function changes with respect to the model parameters.  
A positive gradient means that increasing the parameter increases the loss, and a negative gradient means it decreases.

---

### Updating Parameters

The parameters are updated using the learning rate `α`:

```bash
w = w - α * ∂Loss/∂w
b = b - α * ∂Loss/∂b
```

This process is repeated iteratively over many epochs until convergence.

---

## Key Variance and Concepts to Understand

- **Predicted Variance** refers to the certainty or uncertainty of a prediction.  
  Logistic Regression naturally outputs probabilities that allow interpretation of confidence levels.

- **Cross-Entropy Loss** punishes confident but incorrect predictions more heavily than mild ones.  
  This property makes it ideal for classification tasks where certainty matters.

- A perfectly trained model would have a loss value close to 0, but realistically a very low but non-zero loss is considered good.

---

## Final Summary

- Logistic Regression models the probability of belonging to a class rather than predicting a continuous value.
- It uses a linear function to model the input features and applies the Sigmoid function to ensure output probabilities between 0 and 1.
- It is trained by minimizing Binary Cross-Entropy Loss using Gradient Descent.
- Proper optimization results in the model having low loss values and high classification accuracy.
- Logistic Regression is simple but extremely effective for many binary classification problems.

---

# Important Formulas — Full Cheat Sheet

```bash
# Linear combination of inputs
z = b0 + b1 * x1 + b2 * x2 + ... + bn * xn

# Sigmoid activation
sigmoid(z) = 1 / (1 + exp(-z))

# Predicted probability
y_pred = sigmoid(z)

# Binary Cross-Entropy Loss
Loss = -( y * log(y_pred) + (1 - y) * log(1 - y_pred) )

# Cost function across all examples
J(θ) = -(1/m) * Σ [ y * log(y_pred) + (1 - y) * log(1 - y_pred) ]

# Gradients
∂Loss/∂w = (1/m) * Σ (y_pred - y) * x
∂Loss/∂b = (1/m) * Σ (y_pred - y)

# Gradient Descent Update
w = w - α * ∂Loss/∂w
b = b - α * ∂Loss/∂b
```
