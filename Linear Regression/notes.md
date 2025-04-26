# Main Ideas
- Calculate R²
- Calculate p-value for R²

# Steps
- Draw a line through the data
- Measure the distance from each data point to the line (called a **residual**) and add them up
- Rotate the line slightly
- For each new line, measure the residuals again and compute the **sum of squared residuals**
- Keep rotating, and plot the sum of squared residuals vs rotation angle
- Find the rotation where the sum of squares is the smallest
- Superimpose this "least squares" line on the data  
  (This is your **best fitting line**)

# Least Squares Estimates Two Parameters
- A **y-axis intercept**
- A **slope**

General formula:  
`y = intercept + slope × x`

Example:  
`y = 0.1 + 0.78x`  
(Slope must **not** be zero for the model to make predictions.)

# Variance Concepts
- **Variance** = Sum of Squares / N (where N = number of points)

- **Mean Sum of Squares (SS_Mean)** = Σ(data - mean)²
- **Variance around the mean** = SS_Mean / n

- **Sum of Squares around the fit (SS_Fit)** = Σ(data - line prediction)²
- **Variance around the fit** = SS_Fit / n

# Calculating R²
- R² = (Variance around mean - Variance around fit) / Variance around mean
- Or equivalently:
- R² = (SS_Mean - SS_Fit) / SS_Mean

**Meaning:**  
- R² tells how much of the target's variance is explained by the feature.

If:
- SS_Mean = SS_Fit → R² = 0 (feature explains nothing)

Summary:  
R² = (Variation explained by feature) / (Total variation without feature)

# Calculating p-Value
- **F-statistic** = (Variation explained by feature) / (Variation not explained by the fit)

Detailed formula:

```bash
F = [{SS_Mean - SS_Fit} / (P_Fit - P_Mean)] 
    /
    [SS_Fit / (n - P_Fit)]
```
Where:
- **P_Fit** = number of parameters in the fitted model (usually intercept + slope = 2)
- **P_Mean** = number of parameters for just using the mean line (intercept = 1)

**Idea:**  
- Good fit → F is a **large** number (explained variation is much greater than unexplained)

# Testing p-Value
- Randomly shuffle or sample subsets of the data
- For each subset, calculate F-statistic and plot its histogram
- Repeat the process many times
- Then, for the actual data, plug values into the F formula
- Final p-value = (Number of extreme F-values) / (Total F-values)

# Final Takeaways from Linear Regression
1. Quantify the relationship in the data (**R²**) — needs to be **large**.
2. Assess the reliability of the relationship (**p-value**) — needs to be **small**.

**You need both to trust the model.**
