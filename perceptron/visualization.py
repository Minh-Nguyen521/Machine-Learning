from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
labels = [-1, 1, 1, 1]

def plot_decision_boundary(clf, X, y, title="Perceptron Decision Boundary"):
    # Define the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict on the grid points
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":

    # Prepare data
    X = np.array(list(zip(x1, x2)))
    y = np.array(labels)

    # Train perceptron
    clf = Perceptron(max_iter=10, tol=1e-3, eta0=0.2)
    clf.fit(X, y)

    # Plot decision boundary
    plot_decision_boundary(clf, X, y)