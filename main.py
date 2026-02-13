import numpy as np
import time
import matplotlib.pyplot as plt
import os


SAVE_DIR = "."


# ================================
# PART A – GRADIENT DESCENT
# ================================

def scalar_function(x):
    return x ** 2 + 5


def scalar_gradient(x):
    return 2 * x


def gradient_descent_scalar(x0, alpha, steps=20):
    x = x0
    history = []
    for _ in range(steps):
        history.append(x)
        x = x - alpha * scalar_gradient(x)
    return history


def surface_function(v):
    x, y = v
    return x ** 2 + 3 * y ** 2


def surface_gradient(v):
    x, y = v
    return np.array([2 * x, 6 * y])


def gradient_descent_vector(v0, alpha, steps=20):
    v = v0.copy()
    history = []
    for _ in range(steps):
        history.append(v.copy())
        v = v - alpha * surface_gradient(v)
    return np.array(history)


def plot_trajectory(history):
    """Plot 2D gradient descent trajectory"""
    history = np.array(history)
    plt.figure(figsize=(8, 6))
    plt.plot(history[:, 0], history[:, 1], 'b-o', markersize=4)
    plt.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
    plt.plot(history[-1, 0], history[-1, 1], 'ro', markersize=10, label='End')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title("Gradient Descent Trajectory on f(x,y) = x² + 3y²", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'trajectory_2d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: trajectory_2d.png")


# ================================
# PART B – KNN
# ================================

def generate_dataset(n=100, seed=42):
    np.random.seed(seed)
    class0 = np.random.multivariate_normal([0, 0], np.eye(2), n // 2)
    class1 = np.random.multivariate_normal([2, 2], np.eye(2), n // 2)
    X = np.vstack([class0, class1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def euclidean_distance_loop(x, y):
    # Loop-based implementation
    total = 0.0
    for i in range(len(x)):
        total += (x[i] - y[i]) ** 2
    return np.sqrt(total)


def euclidean_distance_vectorized(x, y):
    # Vectorized implementation - much faster
    return np.linalg.norm(x - y)


def knn_predict(x_test, X_train, y_train, k):
    # Compute all distances using broadcasting
    distances = np.linalg.norm(X_train - x_test, axis=1)

    # Find k nearest neighbors using argpartition (O(n) instead of O(n log n))
    k_indices = np.argpartition(distances, k - 1)[:k]

    # Majority vote
    k_labels = y_train[k_indices]
    prediction = np.bincount(k_labels).argmax()

    return prediction


# ================================
# PERFORMANCE COMPARISON
# ================================

def time_function(func, *args, repeats=1000):
    start = time.time()
    for _ in range(repeats):
        func(*args)
    return time.time() - start


def compare_distance_performance(dim=1000):
    x = np.random.rand(dim)
    y = np.random.rand(dim)

    loop_time = time_function(euclidean_distance_loop, x, y)
    vec_time = time_function(euclidean_distance_vectorized, x, y)

    print(f"Loop-based: {loop_time:.4f}s")
    print(f"Vectorized: {vec_time:.4f}s")
    print(f"Speedup: {loop_time / vec_time:.1f}x")


# ================================
# TASK A1: Scalar GD
# ================================

print("=" * 60)
print("TASK A1: Scalar Gradient Descent")
print("=" * 60)

learning_rates = [0.1, 1.0, 1.1]
plt.figure(figsize=(12, 4))

for idx, alpha in enumerate(learning_rates, 1):
    history = gradient_descent_scalar(10, alpha, 20)

    plt.subplot(1, 3, idx)
    plt.plot(history, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title(f'α = {alpha}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    print(f"\nα = {alpha}:")
    print(f"  Start: {history[0]:.4f}")
    print(f"  End: {history[-1]:.4f}")
    if abs(history[-1]) < 0.5:
        print(f"  Status: CONVERGED ✓")
    elif abs(history[-1]) > 100:
        print(f"  Status: DIVERGED ✗")
    else:
        print(f"  Status: OSCILLATING")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'scalar_gd.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: scalar_gd.png")

# ================================
# TASK A2: 2D Surface GD
# ================================

print("\n" + "=" * 60)
print("TASK A2: 2D Surface Gradient Descent")
print("=" * 60)

v0 = np.array([5.0, 5.0])
history = gradient_descent_vector(v0, 0.1, 20)

print(f"\n{'Step':<6} {'x':<10} {'y':<10} {'f(x,y)':<12}")
print("-" * 40)
for i in [0, 5, 10, 15, 19]:
    x, y = history[i]
    f_val = surface_function(history[i])
    print(f"{i:<6} {x:<10.4f} {y:<10.4f} {f_val:<12.4f}")

print(f"\nObservation: y converges faster than x")
print(f"After 10 iterations: x={history[10][0]:.4f}, y={history[10][1]:.4f}")

# Plot trajectory
plot_trajectory(history)

# Plot convergence over iterations
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history[:, 0], label='x', marker='o', markersize=3)
plt.plot(history[:, 1], label='y', marker='s', markersize=3)
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Convergence: x vs y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
f_values = [surface_function(v) for v in history]
plt.plot(f_values, 'r-o', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('f(x,y)')
plt.title('Function Value Over Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'surface_gd.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: surface_gd.png")

# ================================
# TASK A3: Failure Mode
# ================================

print("\n" + "=" * 60)
print("TASK A3: Failure Mode - Divergence")
print("=" * 60)

history_diverge = gradient_descent_scalar(10, 1.5, 10)

print(f"\n{'Iteration':<12} {'x':<15} {'f(x)':<15}")
print("-" * 42)
for i in range(len(history_diverge)):
    x = history_diverge[i]
    f_val = scalar_function(x)
    print(f"{i:<12} {x:<15.4f} {f_val:<15.2f}")

# Plot divergence
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history_diverge, 'ro-', markersize=5)
plt.xlabel('Iteration')
plt.ylabel('x')
plt.title('Divergence with α = 1.5')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
f_vals = [scalar_function(x) for x in history_diverge]
plt.semilogy(f_vals, 'bo-', markersize=5)
plt.xlabel('Iteration')
plt.ylabel('f(x) (log scale)')
plt.title('Function Value Exploding')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'divergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: divergence.png")

# ================================
# TASK B1: Distance Comparison
# ================================

print("\n" + "=" * 60)
print("TASK B1: Euclidean Distance")
print("=" * 60)

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

dist_loop = euclidean_distance_loop(x, y)
dist_vec = euclidean_distance_vectorized(x, y)

print(f"\nTest vectors: x={x}, y={y}")
print(f"Loop distance: {dist_loop:.6f}")
print(f"Vectorized distance: {dist_vec:.6f}")
print(f"Match: {np.isclose(dist_loop, dist_vec)}")

print("\nPerformance test (1000D vectors, 1000 iterations):")
compare_distance_performance(1000)

# ================================
# TASK B2: Broadcasting
# ================================

print("\n" + "=" * 60)
print("TASK B2: Broadcasting")
print("=" * 60)

X, y = generate_dataset(n=100, seed=42)
x_test = np.array([1.0, 1.0])

# Compute all distances at once
distances = np.linalg.norm(X - x_test, axis=1)

print(f"\nDataset shape: {X.shape}")
print(f"Test point: {x_test}")
print(f"Distances computed: {len(distances)}")
print(f"First 5 distances: {distances[:5]}")

# ================================
# TASK B3: Neighbor Selection
# ================================

print("\n" + "=" * 60)
print("TASK B3: Efficient Neighbor Selection")
print("=" * 60)

k = 5
print(f"\nFinding k={k} nearest neighbors")

# Compare methods
start = time.time()
k_idx_partition = np.argpartition(distances, k - 1)[:k]
time_partition = time.time() - start

start = time.time()
k_idx_sort = np.argsort(distances)[:k]
time_sort = time.time() - start

print(f"argpartition: {time_partition * 1000:.4f} ms")
print(f"argsort: {time_sort * 1000:.4f} ms")
print(f"\nBoth methods find same neighbors: {set(k_idx_partition) == set(k_idx_sort)}")

# ================================
# TASK B4: KNN Prediction
# ================================

print("\n" + "=" * 60)
print("TASK B4: KNN Prediction")
print("=" * 60)

print(f"\n{'Index':<8} {'True':<8} {'Predicted':<10} {'Correct':<10}")
print("-" * 36)

test_indices = [0, 25, 50, 75]
correct = 0

for idx in test_indices:
    x_test = X[idx]
    true_label = y[idx]
    pred = knn_predict(x_test, X, y, k=5)

    is_correct = (pred == true_label)
    if is_correct:
        correct += 1

    check = "✓" if is_correct else "✗"
    print(f"{idx:<8} {true_label:<8} {pred:<10} {check:<10}")

print(f"\nAccuracy: {correct}/{len(test_indices)} = {100 * correct / len(test_indices):.1f}%")

# ================================
# TASK B5: Effect of k
# ================================

print("\n" + "=" * 60)
print("TASK B5: Effect of k Value")
print("=" * 60)

# Split into train/test
X_train, y_train = X[:80], y[:80]
X_test, y_test = X[80:], y[80:]

k_values = [1, 3, 5, 10, 20, 50, 80]
accuracies = []

print(f"\n{'k':<6} {'Accuracy':<12} {'Status':<20}")
print("-" * 38)

for k in k_values:
    correct = 0
    for i in range(len(X_test)):
        pred = knn_predict(X_test[i], X_train, y_train, k)
        if pred == y_test[i]:
            correct += 1

    acc = 100 * correct / len(X_test)
    accuracies.append(acc)

    if k <= 3:
        status = "Overfitting"
    elif k >= 50:
        status = "Underfitting"
    else:
        status = "Good"

    print(f"{k:<6} {acc:<12.1f} {status:<20}")

# Plot effect of k
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('k Value', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Effect of k on KNN Performance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=max(accuracies), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(accuracies):.1f}%')
plt.legend()

plt.subplot(1, 2, 2)
# Visualize dataset
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', label='Class 1', alpha=0.6)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Dataset Visualization', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'knn_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: knn_analysis.png")

# ================================
# PART C: Reflection
# ================================

print("\n" + "=" * 60)
print("PART C: Reflection")
print("=" * 60)

print("\n1. COMPARISON:")
print("   Gradient Descent: Slow training, fast prediction, small memory")
print("   KNN: Fast training, slow prediction, large memory")

print("\n2. SCALABILITY:")
print("   Gradient Descent scales better - prediction is O(d)")
print("   KNN fails for large datasets - prediction is O(n*d)")

print("\n3. WHY KNN RARELY USED IN PRODUCTION:")
print("   - Too slow for real-time (must check all training points)")
print("   - Memory intensive (stores all training data)")
print("   - Curse of dimensionality (fails in high dimensions)")
print("   - No feature learning")

# ================================
# Conclusion
# ================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("\nKey learnings:")
print("  ✓ Learning rate critical for convergence")
print("  ✓ Vectorization gives 100x+ speedup")
print("  ✓ Algorithm choice depends on scale and requirements")
print("  ✓ Understanding fundamentals helps debug and optimize")

print("\n" + "=" * 60)
print("All graphs saved successfully!")
print("=" * 60)