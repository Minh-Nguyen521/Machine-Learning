import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data = [
        [0.204000, 0.834000],
        [0.222000, 0.730000],
        [0.298000, 0.822000],
        [0.450000, 0.842000],
        [0.412000, 0.732000],
        [0.298000, 0.640000],
        [0.588000, 0.298000],
        [0.554000, 0.398000],
        [0.670000, 0.466000],
        [0.834000, 0.426000],
        [0.724000, 0.368000],
        [0.790000, 0.262000],
        [0.824000, 0.338000],
        [0.136000, 0.260000],
        [0.146000, 0.374000],
        [0.258000, 0.422000],
        [0.292000, 0.282000],
        [0.478000, 0.568000],
        [0.654000, 0.776000],
        [0.786000, 0.758000],
        [0.690000, 0.628000],
        [0.736000, 0.786000],
        [0.574000, 0.742000]
    ]

def kmeans_classifier(X_train, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train)
    return kmeans
    
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def visualize_kmeans(X_train, kmeans):
    plt.figure(figsize=(12, 6))
    
    X_train = np.array(X_train)
    
    train_clusters = kmeans.predict(X_train)
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i in range(kmeans.n_clusters):
        cluster_points = X_train[train_clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Cluster {i}', s=50, alpha=0.7)
    
    plt.title('K-Means Clustering (4 Clusters)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print(f"\nKMeans Clustering")
    print("-" * 40)

    data_array = np.array(data)
    n_clusters = 4
    
    kmeans = kmeans_classifier(data_array, n_clusters=n_clusters)
    visualize_kmeans(data_array, kmeans)

if __name__ == "__main__":
    main()
