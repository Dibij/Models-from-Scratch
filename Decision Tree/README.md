# Decision Tree Classifier from Scratch

This repository demonstrates how to implement a **Decision Tree Classifier** from scratch using **Entropy** and **Information Gain** to split nodes in Python.

The goal is to understand the fundamentals of decision trees and how to implement the algorithm without relying on high-level libraries like Scikit-learn.

## Features
- **Binary and Multiclass Classification**: The decision tree works for both binary and multiclass classification tasks.
- **Entropy and Information Gain**: The tree uses entropy and information gain to find the best feature and threshold for splitting.
- **Recursion-Based Tree Growth**: The tree grows recursively, selecting the best splits at each node.
- **Stopping Criteria**: The tree stops growing when certain conditions are met (pure node, max depth, or min samples).

## Requirements

To run the project, you'll need the following libraries:

- `numpy`

You can install it using `pip`:

```bash
pip install numpy scikit-learn 
```

## File Structure

```
decision-tree-classifier-from-scratch/
│
├── decision_tree_classifier.py   # Main script with the decision tree implementation
├── README.md                    # This README file
└── notes.md                   
```

## Usage

### 1. **Training the Model**

The `fit` method trains the decision tree classifier on the input data.

```python
from decision_tree_classifier import DecisionTreeClassifier

# Prepare your dataset (X: features, y: labels)
X = [[2.771244718, 1.784783929],
     [1.728571309, 1.169761413],
     [3.678319846, 2.81281357],
     [3.961043357, 2.619950019],
     [2.999208922, 2.209014212]]

y = [0, 0, 1, 1, 0]

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)

# Fit the classifier to the data
clf.fit(X, y)
```

### 2. **Making Predictions**

Once the model is trained, use the `predict` method to make predictions on new data.

```python
# Predict class labels for new data points
predictions = clf.predict([[3.5, 2.8], [2.8, 2.1]])
print(predictions)
```

### Example

To generate a dataset, train the model, and visualize the results:

```python
import numpy as np
import matplotlib.pyplot as plt
from decision_tree_classifier import DecisionTreeClassifier

# Example data
X = np.array([[2, 3], [3, 3], [4, 3], [1, 2], [2, 1], [3, 1], [4, 2]])
y = np.array([0, 0, 0, 1, 1, 1, 0])

# Train the classifier
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
clf.fit(X, y)

# Predict on new data
predictions = clf.predict([[3, 2], [2, 2]])

# Print predictions
print(predictions)

# Visualize the decision tree (if needed, you can extend this visualization)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
