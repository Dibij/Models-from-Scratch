## Decision Tree Classifier — In-Depth Notes (Mathematics + Code Formulas)

---

## Main Concepts

A **Decision Tree** is a **supervised machine learning algorithm** used for both classification and regression tasks. It works by recursively splitting the data into subsets based on feature values, ultimately creating a tree-like structure of decision nodes and leaf nodes.

The core idea is to split the data in a way that results in **pure child nodes** (nodes that contain data of a single class in classification problems). The algorithm chooses the best feature and threshold to split based on **information gain** (for classification tasks).

---

## Workflow Steps

---

### Step 1: Select the Best Split

To build the tree, the decision tree algorithm first selects a feature and a threshold to split the data. The goal is to split the dataset such that the resulting subsets (or child nodes) are as **pure** as possible.

```bash
Best Split: Feature idx, Threshold value
```

For each feature in the dataset:
- The algorithm considers all unique values of the feature as possible thresholds.
- The **information gain** is calculated for each threshold. The split with the highest information gain is chosen.

---

### Step 2: Compute Information Gain

Information Gain is calculated using **Entropy**. The entropy of a set quantifies the uncertainty or impurity of the set. A pure set (all elements belonging to the same class) has zero entropy.

```bash
Entropy(D) = - Σ [ p_i * log2(p_i) ]
```
where `p_i` is the proportion of class `i` in the set `D`.

After a split, the **weighted average** of the entropy of the child nodes is used to calculate the **Information Gain**.

```bash
Information Gain = Entropy(Parent) - Weighted Sum of Entropy(Children)
```

---

### Step 3: Recursively Build the Tree

The tree is recursively built by splitting the dataset and applying the same procedure to the resulting subsets. This process continues until one of the stopping conditions is met:
1. The node reaches a **pure state** (entropy is 0).
2. The tree reaches the **maximum depth** (predefined in the hyperparameters).
3. The node contains **less than the minimum number of samples** to split.

---

## Mathematical Details

---

### Entropy

Entropy is used to measure the impurity of a set of labels. The goal is to split the dataset such that each child node has lower entropy.

For a dataset with two classes, the entropy formula is:

```bash
Entropy(D) = - ( p_class1 * log2(p_class1) + p_class2 * log2(p_class2) )
```
where `p_class1` and `p_class2` are the proportions of class 1 and class 2 in the dataset `D`.

---

### Information Gain

Information Gain measures the reduction in entropy after a split:

```bash
Information Gain = Entropy(D) - ( (n_left / n) * Entropy(D_left) + (n_right / n) * Entropy(D_right) )
```

- `n_left` and `n_right` are the number of samples in the left and right child nodes.
- `n` is the total number of samples in the parent node.

---

### Decision Node Class

Each decision node represents a feature-based split in the dataset. The node stores:
- `feature_idx`: Index of the feature used to split.
- `threshold`: Threshold value for the split.
- `left`: Left child node.
- `right`: Right child node.
- `value`: Class label if the node is a leaf.

---

### Stopping Criteria

1. **Maximum Depth**: The tree will not grow beyond a certain depth (`max_depth`).
2. **Minimum Samples Split**: If a node contains fewer than `min_samples_split` samples, it will not be split further.
3. **Pure Node**: If a node's dataset is perfectly pure (all samples belong to the same class), the tree stops splitting further at that node.

---

## Key Variance and Concepts to Understand

- **Purity**: A node is considered pure when all samples at the node belong to the same class.
- **Overfitting**: Decision trees are prone to overfitting, especially with deeper trees. Pruning and regularization techniques are often used to prevent this.
- **Entropy and Gini Index**: The entropy-based method is one way to measure the impurity of a node. Another common method is the Gini Index, but the code uses entropy.

---

## Final Summary

- **Decision Trees** are powerful, intuitive classifiers that work by splitting the dataset based on features.
- **Entropy** is used to measure the impurity of nodes, and **Information Gain** is used to select the best feature and threshold for splits.
- The algorithm stops growing the tree when certain stopping criteria are met, like reaching a certain depth or creating pure child nodes.
- **Overfitting** can be a problem, but pruning and hyperparameter tuning can help mitigate it.

---

# Important Formulas — Full Cheat Sheet

```bash
# Entropy
Entropy(D) = - Σ [ p_i * log2(p_i) ]

# Information Gain
Information Gain = Entropy(Parent) - Weighted Sum of Entropy(Children)

# Information Gain Calculation (for a particular feature and threshold)
Information Gain = Entropy(D) - ( (n_left / n) * Entropy(D_left) + (n_right / n) * Entropy(D_right) )

# Decision Node Class
DecisionNode:
    feature_idx = index of feature used for split
    threshold = threshold value for split
    left = left child node
    right = right child node
    value = class label if leaf node
```

