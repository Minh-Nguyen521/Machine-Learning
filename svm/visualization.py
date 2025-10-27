import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd

def create_sample_data():
    """Create the sample data from the table"""
    data = [
        [0.204000, 0.834000, 0],
        [0.222000, 0.730000, 0],
        [0.298000, 0.822000, 0],
        [0.450000, 0.842000, 0],
        [0.412000, 0.732000, 0],
        [0.298000, 0.640000, 0],
        [0.588000, 0.298000, 0],
        [0.554000, 0.398000, 0],
        [0.670000, 0.466000, 0],
        [0.834000, 0.426000, 0],
        [0.724000, 0.368000, 0],
        [0.790000, 0.262000, 0],
        [0.824000, 0.338000, 0],
        [0.136000, 0.260000, 1],
        [0.146000, 0.374000, 1],
        [0.258000, 0.422000, 1],
        [0.292000, 0.282000, 1],
        [0.478000, 0.568000, 1],
        [0.654000, 0.776000, 1],
        [0.786000, 0.758000, 1],
        [0.690000, 0.628000, 1],
        [0.736000, 0.786000, 1],
        [0.574000, 0.742000, 1]
    ]
    
    df = pd.DataFrame(data, columns=['X1', 'X2', 'Class'])
    X = df[['X1', 'X2']].values
    y = df['Class'].values
    
    return X, y

def plot_svm_simple(X, y, kernel='rbf', C=10):
    # Train SVM classifier
    svm_clf = SVC(kernel=kernel, C=C, gamma='scale')
    svm_clf.fit(X, y)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create mesh for decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for the mesh
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['lightblue', 'lightyellow'])
    
    # Plot decision boundary
    Z_boundary = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_boundary = Z_boundary.reshape(xx.shape)
    plt.contour(xx, yy, Z_boundary, levels=[0], colors='green', linewidths=2)
    
    # Plot data points
    colors = ['blue', 'red']
    labels = ['Class 0', 'Class 1']
    for i in range(2):
        mask = (y == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=80, 
                   label=labels[i], edgecolors='black', linewidth=1)
    
    # Labels and title
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)
    plt.title('SVC Decision Boundary', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return svm_clf

def main():
    # Load data
    X, y = create_sample_data()
    
    # Create simple SVM plot
    svm_clf = plot_svm_simple(X, y, kernel='rbf', C=10)
    plt.show()
    
    # Print basic info
    print(f"Number of support vectors: {len(svm_clf.support_vectors_)}")
    print(f"Kernel: RBF")
    print(f"C parameter: 10")

if __name__ == "__main__":
    main()