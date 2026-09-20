# Data Classification Using AI — Project 2

Internship project for **DecodeLabs** (Industrial Training Kit, Batch 2026).

## Overview

A supervised learning classification model built in Python using the classic Iris dataset. The goal is to teach a machine to recognize patterns in flower measurements and correctly categorize new samples into one of three species.

This project moves beyond fixed rules (Project 1) into real machine learning: the model isn't told the rules directly — it learns a decision boundary from labeled training data (the Iris dataset), then is tested on unseen data to check how well it generalizes.

## Dataset

The Iris dataset — 150 samples, 3 balanced classes (Setosa, Versicolor, Virginica), 4 numeric features per sample:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Pipeline

1. **Load the dataset** — via `sklearn.datasets.load_iris()`
2. **Train-test split** — 80% training, 20% testing, stratified to keep class balance
3. **Feature scaling** — `StandardScaler` normalizes all features to mean 0, variance 1, since KNN relies on distance calculations
4. **Model** — K-Nearest Neighbors (`KNeighborsClassifier`) with `k=7`, chosen after testing k values 1-20 for the lowest error rate
5. **Evaluation** — accuracy score, confusion matrix, and a full classification report (precision, recall, F1-score per class)

## How to Run

```bash
python Project-02.py
```

## Example Output

```
Dataset shape: (150, 4)
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Classes: ['setosa', 'versicolor', 'virginica']

Training samples: 120
Testing samples: 30

Accuracy: 96.67 %

Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  1  9]]

Classification Report (Precision / Recall / F1):
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.91      1.00      0.95        10
   virginica       1.00      0.90      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

## Tech Used

- Python 3
- scikit-learn

## Author

Part of the DecodeLabs AI Engineering Internship — Batch 2026.
