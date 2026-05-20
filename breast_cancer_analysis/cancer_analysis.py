# ==============================
# Breast Cancer ML Analysis
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import svd


# -------------------------------------------------
# NOTE:
# You must already have these functions defined:
# SGD_gradient_descent()
# gradient()
# MSE_cost()
# -------------------------------------------------


# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv('cancer_data.csv', header=None)

X = data.drop(columns=data.columns[-1])
y = data[data.columns[-1]].values

X = X.astype('float64')
y = y.astype('float64')


# ==============================
# 2. Normalize Features
# ==============================
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)


# ==============================
# 3. Add Bias Column
# ==============================
X_normalized_with_bias = X_normalized.copy()
X_normalized_with_bias.insert(0, 'bias', 1)


# Rename columns
X_normalized_with_bias.columns = list(range(X_normalized_with_bias.shape[1]))


# ==============================
# 4. Gradient Descent Comparison
# ==============================
iterations = 100
alphas = [0.1, 0.01, 0.001]

plt.figure(figsize=(6, 5))

for alpha in alphas:
    theta = np.zeros(X_normalized_with_bias.shape[1])
    _, cost = SGD_gradient_descent(
        X_normalized_with_bias,
        y,
        theta,
        alpha,
        iterations
    )

    plt.plot(range(iterations), cost, label=f"α = {alpha}")

plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("SGD Convergence for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# 5. Mini-Batch Gradient Descent
# ==============================
def mini_batch_gradient_descent(X, y, theta, alpha, iterations, batch_size):
    m = len(y)

    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y_np = y.to_numpy() if isinstance(y, pd.Series) else y

    cost_history = []
    n_batches = int(m / batch_size)

    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X_np[indices]
        y_shuffled = y_np[indices]

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            theta = theta - alpha * gradient(theta, X_batch, y_batch)

        cost = MSE_cost(theta, X, y)
        cost_history.append(cost)

    return theta, cost_history


# ==============================
# 6. Run Mini-Batch Experiments
# ==============================
batch_size = 32
iterations_mb = 100

plt.figure(figsize=(6, 5))

for alpha in alphas:
    theta = np.zeros(X_normalized_with_bias.shape[1])

    _, cost_history = mini_batch_gradient_descent(
        X_normalized_with_bias,
        y,
        theta,
        alpha,
        iterations_mb,
        batch_size
    )

    plt.plot(cost_history, label=f"α = {alpha}")

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Mini-Batch Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# 7. SVD Dimensionality Reduction
# ==============================
def reduce_dimensions_svd(X, n_components=3):
    U, S, VT = svd(X, full_matrices=False)
    return np.dot(U[:, :n_components], np.diag(S[:n_components]))


X_reduced = reduce_dimensions_svd(X_normalized, n_components=3)

X_reduced_with_bias = np.c_[np.ones((X_reduced.shape[0], 1)), X_reduced]


# ==============================
# 8. Mini-Batch on Reduced Data
# ==============================
plt.figure(figsize=(6, 5))

for alpha in alphas:
    theta = np.zeros(X_reduced_with_bias.shape[1])

    _, cost_history = mini_batch_gradient_descent(
        X_reduced_with_bias,
        y,
        theta,
        alpha,
        iterations_mb,
        batch_size
    )

    plt.plot(cost_history, label=f"α = {alpha}")

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Mini-Batch (Reduced Data)")
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# 9. Runtime Comparison
# ==============================
runtimes_full = []
runtimes_reduced = []

for alpha in alphas:
    theta = np.zeros(X_normalized_with_bias.shape[1])

    start = time.time()
    SGD_gradient_descent(X_normalized_with_bias, y, theta, alpha, iterations)
    end = time.time()

    runtimes_full.append(end - start)


for alpha in alphas:
    theta = np.zeros(X_reduced_with_bias.shape[1])

    start = time.time()
    SGD_gradient_descent(X_reduced_with_bias, y, theta, alpha, iterations)
    end = time.time()

    runtimes_reduced.append(end - start)


print("Full data runtime:", runtimes_full)
print("Reduced data runtime:", runtimes_reduced)


# ==============================
# 10. Mini-Batch Runtime Comparison
# ==============================
runtimes_full_mb = []
runtimes_reduced_mb = []

for alpha in alphas:
    theta = np.zeros(X_normalized_with_bias.shape[1])

    start = time.time()
    mini_batch_gradient_descent(X_normalized_with_bias, y, theta, alpha, iterations_mb, batch_size)
    end = time.time()

    runtimes_full_mb.append(end - start)


for alpha in alphas:
    theta = np.zeros(X_reduced_with_bias.shape[1])

    start = time.time()
    mini_batch_gradient_descent(X_reduced_with_bias, y, theta, alpha, iterations_mb, batch_size)
    end = time.time()

    runtimes_reduced_mb.append(end - start)


print("Mini-Batch Full runtime:", runtimes_full_mb)
print("Mini-Batch Reduced runtime:", runtimes_reduced_mb)