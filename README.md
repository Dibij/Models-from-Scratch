# Linear Regression from Scratch

This repository demonstrates how to implement **Linear Regression** from scratch using **Gradient Descent** and **SSR Minimization** in Python.

The goal is to understand the fundamentals of linear regression and how to implement it without relying on any high-level libraries like Scikit-learn. The project includes two methods for fitting the best line to data:

- **SSR Minimization** (Sum of Squared Residuals)
- **Gradient Descent**

## Features
- Generate synthetic data to simulate a linear relationship.
- Implement a brute-force approach using SSR minimization to find the best fit line.
- Implement Gradient Descent to optimize the linear regression parameters.
- Plot the data points, the best fit line from both methods, and the cost history for gradient descent.

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
linearregression-from-scratch/
│
├── linear_regression.py        # Main script with the implementation
├── README.md                  # This README file
└── data/                      # Folder to store data (if required)
```

## Usage

### 1. **SSR Minimization Method**
This method finds the best slope and intercept by brute-force search, minimizing the Sum of Squared Residuals (SSR).

```python
best_slope, best_intercept, R_squared = best_fit_ssr(x, y)
```

### 2. **Gradient Descent Method**
This method uses gradient descent to minimize the cost function (MSE) and optimize the parameters.

```python
m, b, cost_history = gradient_descent(x, y, alpha=0.1, iterations=1000)
```

### Example
To generate data, fit the model, and visualize the results:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)

# SSR Minimization
best_slope, best_intercept, R_squared = best_fit_ssr(x, y)

# Gradient Descent
m, b, cost_history = gradient_descent(x, y)

# Visualize the results
plt.scatter(x, y, label='Data Points')
plt.plot(x, best_slope * x + best_intercept, color='red', label='Best Fit Line (SSR Minimization)')
plt.plot(x, m * x + b, color='green', label='Best Fit Line (Gradient Descent)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
