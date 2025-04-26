# Models from Scratch

This repository contains simple, from-scratch implementations of three fundamental machine learning models:

- **Linear Regression**
- **Logistic Regression**
- **Decision Tree Classifier**

Built without any machine learning libraries like Scikit-learn — only using **NumPy** and **Matplotlib**.

---

## Linear Regression

- **Goal:** Predict a continuous value.
- **Method:** Minimize the **sum of squared residuals** between predicted and actual values.
- **Metrics Used:**
  - **R² (Coefficient of Determination):**  
    Measures how much of the variance in the target variable is explained by the feature.
  - **p-value:**  
    Evaluates the statistical significance of the relationship between the feature and the target.
- **Important Concepts:**
  - Drawing different lines through the data.
  - Measuring residuals and finding the line that minimizes total squared error.
  - Calculating R² and p-values to judge both the **quality** and **trustworthiness** of the model.

---

## Logistic Regression

- **Goal:** Classify data into two classes (binary classification).
- **Method:** 
  - Apply a **sigmoid function** to model the probability that a data point belongs to class 1.
  - Optimize weights by minimizing the **logistic loss** using **gradient descent**.
- **Metrics Used:**
  - **Accuracy** on training data.
- **Important Concepts:**
  - Adding a bias term (intercept) to the features.
  - Learning the weights through iterative optimization.
  - Visualizing both the **training process** (cost vs iterations) and the **final decision boundary**.
  - Making actual probability predictions for new, unseen points.

---

## Decision Tree Classifier

- **Goal:** Classify data into two or more classes (binary or multiclass classification).
- **Method:** 
  - Split the dataset recursively based on **Entropy** and **Information Gain** to build a tree structure.
  - At each node, choose the feature and threshold that result in the purest possible child nodes.
- **Metrics Used:**
  - **Entropy** to measure the disorder or impurity at a node.
  - **Information Gain** to quantify the effectiveness of a split.
- **Important Concepts:**
  - Growing the tree recursively by selecting the best splits.
  - Stopping conditions like max depth, minimum samples at a node, or perfect purity.
  - Understanding how a decision tree essentially creates a set of logical if-else conditions to classify points.

---

## Final Thoughts

All models are implemented from the ground up without any shortcuts, ensuring a deep understanding of how they actually work internally — not just calling `.fit()` and `.predict()`.

---
**Feel free to explore the code blocks and plots to get a better sense of how real models are built piece by piece.**
