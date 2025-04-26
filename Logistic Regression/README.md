# Logistic Regression from Scratch

This repository demonstrates how to implement **Logistic Regression** from scratch using **Gradient Descent** in Python.

The goal is to understand the fundamentals of logistic regression and how to implement it without relying on high-level libraries like Scikit-learn. The project includes the following steps:

- Implement **Sigmoid Function** for classification.
- Implement **Cost Function** based on **Log-Loss**.
- Use **Gradient Descent** to optimize the parameters and minimize the cost function.
- Visualize the synthetic data, decision boundary, and cost history.

## Features
- Generate synthetic data for binary classification.
- Implement **Sigmoid** and **Cost Function** (Log-Loss).
- Use **Gradient Descent** to find the optimal parameters.
- Visualize the decision boundary and the cost history during optimization.

## Requirements

To run the project, you'll need the following libraries:

- `numpy`
- `matplotlib`

You can install them using `pip`:

```bash
pip install numpy matplotlib
```

## File Structure

```
logistic-regression-from-scratch/
│
├── logistic_regression.py        # Main script with the implementation
├── README.md                     # This README file
└── data/                         # Folder to store data (if required)
```

## Usage

### 1. **Sigmoid Function**
The **sigmoid function** is used to map the output to a probability between 0 and 1, making it ideal for binary classification.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### 2. **Cost Function (Log-Loss)**
The **log-loss** cost function is used to measure how well the logistic regression model's predictions match the actual outcomes.

```python
def compute_cost(X, y, theta):
    m = len(y)
    z = X.dot(theta)
    predictions = sigmoid(z)
    cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost
```

### 3. **Gradient Descent**
This method uses gradient descent to minimize the cost function by adjusting the weights iteratively.

```python
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = sigmoid(X.dot(theta))
        theta -= (alpha / m) * X.T.dot(predictions - y)
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history
```

### Example
To generate data, fit the logistic regression model, and visualize the results:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Class 1 if sum of features > 0 else Class 0

# Add a column of ones to X to account for the intercept term
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize parameters
theta_initial = np.zeros(X_bias.shape[1])  # Initialize weights to zeros
alpha = 0.1  # Learning rate
iterations = 1000  # Number of iterations

# Train the model using Gradient Descent
theta_optimal, cost_history = gradient_descent(X_bias, y, theta_initial, alpha, iterations)

# Visualize the cost history
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function History')
plt.show()

# Predict using the optimal theta
predictions = sigmoid(X_bias.dot(theta_optimal)) >= 0.5

# Visualize the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Predictions')
plt.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
