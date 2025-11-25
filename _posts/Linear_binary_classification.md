---
title: 'Future Blog Post'
date: 2199-01-01
permalink: /posts/2012/08/blog-post-6/
tags:
  - cool posts
  - category1
  - category2
---

## Introduction
Supervised learning is a branch of machine learning in which an algorithm is trained on labelled input–output pairs. The goal is to learn a mapping from inputs to outputs that generalises well to unseen data. When the outputs belong to a small, finite set of categories, the task is called **classification**. When the outputs come from an ordered or continuous set, the task is known as **regression**.

This note focuses on the **perceptron algorithm**, a foundational method for solving linear binary classification problems.

<hr style="height: 1px;">

## Linear Binary Classification
Consider a training set 
\begin{equation}
\left\{ \left({\mathbf{x}}_i, y_i\right) \right\}^N_{i=1},
\end{equation} 
where each ${\mathbf{x}}_i$ is a feature vector and each label $y_i$ takes values in $\left\lbrace-1, +1\right\rbrace$. The goal is to find a linear function that maps feature vectors to labels. We consider classifiers of the form
\begin{equation}
f\left({\mathbf{x}}\right) = {\mathbf{w}}^{\top}{{\mathbf{x}}}+ w_0=\hat{\mathbf{w}}^{\top}\hat{\mathbf{x}},
\tag{1}\end{equation}
where $\hat{\mathbf{x}}=\left[{\mathbf{x}}, 1\right]^{\top}$ and $\hat{\mathbf{w}}=\left[{\mathbf{w}}, w_0\right]^{\top}$ is a vector of real-valued parameters. Different choices of $\hat{\mathbf{w}}$ produce different linear functions, each corresponding to a hyperplane that attempts to separate the two classes in the training data.

To evaluate a particular classifier, we measure its **training error**:
\begin{equation}
\frac{1}{N}\sum_{i=1}^N \mathbf{1}\!\left[f(\mathbf{x}_i) y_i \le 0 \right],
\tag{2}\end{equation}
where the indicator function $\mathbf{1}[\cdot]$ equals $1$ if the condition is true and $0$ otherwise. This quantity counts the proportion of misclassified training examples.

<hr style="height: 1px;">

## Perceptron Algorithm
The perceptron algorithm seeks a parameter vector $\hat{\mathbf{w}}$ that minimises the number of misclassifications. A training example $\left({\mathbf{x}}_i, y_i\right)$ is considered **misclassified** when
\begin{equation}
y_i\hat{\mathbf{w}}^\top\hat{\mathbf{x}}_i\leq 0.
\tag{3}\end{equation}

The algorithm iterates through the training set and updates the parameters only when a misclassification occurs. The update rule is
\begin{equation}
{\hat{\mathbf{w}}}\gets \hat{\mathbf{w}}+ y_i\hat{\mathbf{x}}_i. 
\tag{4}\end{equation}
To see why this corrects the mistake, consider evaluating the classifier on the same input after an update:
\begin{equation}
y_i\hat{\mathbf{w}}^\top\hat{\mathbf{x}}_i\quad \text{increases by}\quad \lVert{\hat{\mathbf{x}}_i\rVert}^2.
\tag{5}\end{equation}
Thus, repeated updates push the classifier toward correctly classifying ${\mathbf{x}}_i$. Misclassification of other examples may push $\hat{\mathbf{w}}$ in different directions, but the algorithm stops once all examples are correctly classified.

If the data are linearly separable, the perceptron algorithm converges after a finite number of updates. For proof of convergence, see, e.g., Jaakkola (2006).

<hr style="height: 1px;">

## Learning Algorithm: Perceptron
**Initialize**:
\begin{equation}
\hat{\mathbf{w}}=\mathbf{0}
\end{equation}

**Repeat until all examples are correctly classified**:
1. For each training example $\left({\mathbf{x}}_i, y_i\right)$:
   - if $y_i\hat{\mathbf{w}}^\top\hat{\mathbf{x}}_i\leq 0$, update
     \begin{equation}{\hat{\mathbf{w}}}\gets \hat{\mathbf{w}}+ y_i\hat{\mathbf{x}}_i\end{equation}
     
<hr style="height: 1px;">

## Implementation with NumPy


```python
import numpy as np

def perceptron_train(X, y, max_epochs=1000):
    """
    Train a perceptron classifier.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training feature vectors.
    y : ndarray of shape (n_samples,)
        Class labels in {-1, +1}.
    max_epochs : int, optional
        Maximum number of passes over the training set.

    Returns
    -------
    w : ndarray of shape (n_features,)
        Learned weight vector.
    """
    n_samples, n_features = X.shape

    # Initialize weights to zero
    w = np.zeros(n_features, dtype=float)

    for epoch in range(max_epochs):
        errors = 0

        for i in range(n_samples):
            # Check if the current example is misclassified
            if y[i] * np.dot(w, X[i]) <= 0:
                # Update rule: w <- w + y_i * x_i
                w += y[i] * X[i]
                errors += 1

        # Stop if there are no misclassifications in this epoch
        if errors == 0:
            break

    return w


def perceptron_predict(X, w):
    """
    Predict labels for input data using a trained perceptron.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input feature vectors.
    w : ndarray of shape (n_features,)
        Weight vector learned by perceptron_train.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted labels in {-1, +1}.
    """
    scores = X @ w
    # Map scores to {-1, +1}
    return np.where(scores >= 0, 1, -1)
```

Example application need to come here


```python
# X: (n_samples, n_features), y: labels in {-1, +1}
w = perceptron_train(X, y)
y_pred = perceptron_predict(X, w)
```
