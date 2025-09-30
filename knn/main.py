import pandas as pd
from collections import Counter
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def get_neighbors(X_train, y_train, test_instance, k):
    distances = np.sum(np.abs(X_train - test_instance), axis=1)

    k_nearest_indices = np.argpartition(distances, k)[:k]
    neighbors = y_train[k_nearest_indices]
    return neighbors

def predict_classification(neighbors):
    vote_counts = Counter(neighbors)
    return vote_counts.most_common(1)[0][0]

def knn_classifier(X_train, y_train, X_test, k):
    predictions = []
    total_samples = len(X_test)
    
    for i in range(total_samples):            
        test_instance = X_test[i]
        neighbors = get_neighbors(X_train, y_train, test_instance, k)
        prediction = predict_classification(neighbors)
        predictions.append(prediction)
    
    return predictions

def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)

def create_confusion_matrix(y_true, y_pred):
    classes = sorted(set(list(y_true) + list(y_pred)))
    n_classes = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    
    for i in range(len(y_true)):
        true_idx = class_to_idx[y_true[i]]
        pred_idx = class_to_idx[y_pred[i]]
        matrix[true_idx][pred_idx] += 1
    
    return matrix, classes

def print_confusion_matrix(matrix, classes):
    print("\nConfusion Matrix:")
    print("Predicted:", end="")
    for cls in classes:
        print(f"{cls:>6}", end="")
    print()
    
    for i, cls in enumerate(classes):
        print(f"Actual {cls}:", end="")
        for j in range(len(classes)):
            print(f"{matrix[i][j]:>6}", end="")
        print()

def evaluate_dataset(train_file, test_file, dataset_name):
    print(f"Dataset: {dataset_name}")
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
        
    k_values = [1, 3, 5]    
    for k in k_values:
        print(f"k = {k}")
        
        predictions = knn_classifier(X_train, y_train, X_test, k)
        
        accuracy = calculate_accuracy(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        conf_matrix, classes = create_confusion_matrix(y_test, predictions)
        print_confusion_matrix(conf_matrix, classes)

def main():
    datasets = [
        ('data/iris/iris.trn', 'data/iris/iris.tst', 'Iris'),
        ('data/optics/opt.trn', 'data/optics/opt.tst', 'Optics'),
        ('data/letter/let.trn', 'data/letter/let.tst', 'Letter'),
        ('data/faces/data.trn', 'data/faces/data.tst', 'Face'),
        ('data/fp/fp.trn', 'data/fp/fp.tst', 'Fp')
    ]
    
    for train_file, test_file, name in datasets:
        evaluate_dataset(train_file, test_file, name)
        

if __name__ == "__main__":
    main()
