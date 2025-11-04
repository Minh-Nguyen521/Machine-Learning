import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def kmeans_classifier(X_train, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_train)

    return kmeans

def visualize_kmeans(X_train, kmeans, dataset_name):
    train_clusters = kmeans.predict(X_train)
    
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    df = pd.DataFrame(X_train, columns=feature_names)
    df['Cluster'] = train_clusters
    
    sns.pairplot(df, hue='Cluster', palette=['blue', 'green', 'red'])

    plt.tight_layout()
    plt.show()

def evaluate_dataset(train_file, test_file, dataset_name):
    X_train, _ = load_data(train_file)
        
    n_clusters = 3
    kmeans = kmeans_classifier(X_train, n_clusters=n_clusters)

    visualize_kmeans(X_train, kmeans, dataset_name)

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
