import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 1. Define data
X = np.array([[0, 0], [0, 1], [1, 0], [1.1, 1]])
y = np.array([-1, 1, 1, 1])

# 2. Train the SVM
clf = svm.SVC(kernel='linear', C=10)
clf.fit(X, y)

# 3. Get hyperplane parameters
w = clf.coef_[0]
b = clf.intercept_[0]
slope = -w[0] / w[1]
y_intercept = -b / w[1]

# # 4. Print model details
# print(f"Weights (w): {w}")
# print(f"Intercept (b): {b[0]:.4f}")
# print(f"Slope (a): {slope:.4f}")
# print(f"Y-Intercept (c): {y_intercept:.4f}")
# print(f"Support Vectors:\n{clf.support_vectors_}")

# 5. Plotting
plt.figure(figsize=(10, 9))
xx = np.linspace(-0.5, 1.5)

# Plot decision boundary
yy = slope * xx + y_intercept
plt.plot(xx, yy, 'k-', label='Decision Boundary')

# Plot upper margin (w.x + b = 1)
yy_upper = slope * xx + (-b + 1) / w[1]
plt.plot(xx, yy_upper, 'r--', label='Upper Margin')

# Plot lower margin (w.x + b = -1)
yy_lower = slope * xx + (-b - 1) / w[1]
plt.plot(xx, yy_lower, 'b--', label='Lower Margin')

# Plot data points
plt.scatter(X[y == -1, 0], X[y == -1, 1], s=200, marker='s', c='blue', label='Class -1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=200, marker='o', c='red', label='Class +1')

# Highlight Support Vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='k', linewidths=1.5,
            label='Support Vectors')

# 6. Customize plot
plt.title('SVM Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True)
plt.legend(loc='upper right')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()