import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def kmeans_classifier(X_train, y_train, X_test, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train)
    
    cluster_predictions = kmeans.predict(X_test)
    
    cluster_labels = kmeans.predict(X_train)
    label_mapping = {}
    
    for cluster in range(n_clusters):
        cluster_mask = (cluster_labels == cluster)
        if np.any(cluster_mask):
            most_common_label = np.bincount(y_train[cluster_mask]).argmax()
            label_mapping[cluster] = most_common_label
        else:
            label_mapping[cluster] = 0 
    
    predictions = np.array([label_mapping.get(cluster, 0) for cluster in cluster_predictions])
    
    return predictions, kmeans, cluster_predictions

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def visualize_kmeans(X_train, y_train, X_test, kmeans, cluster_predictions, dataset_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    train_clusters = kmeans.predict(X_train)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=train_clusters, cmap='tab10', s=50, alpha=0.7)
    
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, linewidths=2, 
                edgecolors='black', label='Centroids')
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'{dataset_name} - Training Data (Colored by Cluster ID)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Test data with predicted clusters
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=cluster_predictions, cmap='tab10', s=50, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, linewidths=2, 
                edgecolors='black', label='Centroids')
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'{dataset_name} - Test Data (Colored by Cluster ID)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def print_confusion_matrix(y_true, y_pred):
    # Get unique classes and create confusion matrix
    classes = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    print(cm_df)

def evaluate_dataset(train_file, test_file, dataset_name):
    print(f"Dataset: {dataset_name}")
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
        
    print(f"\nKMeans Classification")
    print("-" * 40)

    n_clusters = 3
    predictions, kmeans, cluster_predictions = kmeans_classifier(X_train, y_train, X_test, n_clusters=n_clusters)
    accuracy = calculate_accuracy(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print_confusion_matrix(y_test, predictions)
    
    visualize_kmeans(X_train, y_train, X_test, kmeans, cluster_predictions, dataset_name)

def main():
    datasets = [
        ('data/iris/iris.trn', 'data/iris/iris.tst', 'Iris'),
        # ('data/optics/opt.trn', 'data/optics/opt.tst', 'Optics'),
        # ('data/letter/let.trn', 'data/letter/let.tst', 'Letter'),
        # ('data/leukemia/ALLAML.trn', 'data/leukemia/ALLAML.tst', 'Leukemia'),
        # ('data/fp/fp.trn', 'data/fp/fp.tst', 'Fp')
    ]
    
    for train_file, test_file, name in datasets:
        evaluate_dataset(train_file, test_file, name)

if __name__ == "__main__":
    main()
