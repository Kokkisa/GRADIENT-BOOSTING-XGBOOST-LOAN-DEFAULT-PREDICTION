[README_project10.md](https://github.com/user-attachments/files/25766920/README_project10.md)
# 🏦 Gradient Boosting & XGBoost — Loan Default Prediction

## Overview
Classification project comparing five models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) to predict loan defaults. Demonstrates the evolution from linear models to ensemble methods, hyperparameter tuning with GridSearchCV, and business impact analysis on a 3,000-loan dataset.

**Built by:** Nithin Kumar Kokkisa — Senior Demand Planner with 12+ years in manufacturing operations & supply chain analytics.

---

## Business Problem
A lending company needs to identify borrowers likely to default on their loans. False negatives (missing a default) result in significant financial losses. False positives (rejecting good borrowers) result in lost interest revenue. This project builds models that balance both risks and quantifies the financial impact of each error type.

## Approach

### Feature Engineering
- Encoded categorical variables (Home Ownership, Loan Purpose)
- Created: Loan_to_Income ratio, Credit_x_Income interaction
- Handled class imbalance with scale_pos_weight

### Five Models Compared (Evolution)
1. **Logistic Regression** — Linear baseline
2. **Decision Tree** — Non-linear, single tree
3. **Random Forest** — Bagging (parallel trees, majority vote)
4. **Gradient Boosting** — Boosting (sequential trees, error correction)
5. **XGBoost** — Optimized boosting with regularization

### Hyperparameter Tuning
- GridSearchCV with 3-fold cross-validation
- Tuned: max_depth, learning_rate, n_estimators

## Key Results

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | TBD | TBD |
| Decision Tree | TBD | TBD |
| Random Forest | TBD | TBD |
| Gradient Boosting | TBD | TBD |
| XGBoost | TBD | TBD |

## Concepts Demonstrated
- Decision Trees (Gini impurity, recursive splitting)
- Bagging vs Boosting (variance reduction vs bias reduction)
- Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning (GridSearchCV)
- ROC curves and AUC comparison
- Class imbalance handling
- Business impact analysis (cost of errors)

## Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **scikit-learn** (DecisionTree, RandomForest, GradientBoosting, GridSearchCV)
- **XGBoost**

---

## About
Part of a **30-project data analytics portfolio**. See [GitHub profile](https://github.com/Kokkisa) for the full portfolio.
