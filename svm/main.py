import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def svm_classifier(X_train, y_train, X_test, kernel='rbf', C=1.0, gamma='scale', random_state=42):
    svm_clf = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
    svm_clf.fit(X_train, y_train)
    predictions = svm_clf.predict(X_test)
    
    return predictions, svm_clf

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def print_confusion_matrix(y_true, y_pred):
    # Get unique classes and create confusion matrix
    classes = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print("\nConfusion Matrix:")
    # Create a pandas DataFrame for cleaner display

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    print(cm_df)
    
def evaluate_dataset(train_file, test_file, dataset_name):
    print(f"Dataset: {dataset_name}")
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
        
    best_accuracy = 0
    best_predictions = None
    best_params = None
    
    print(f"\nSVM Classification")
    print("-" * 40)

    # SVM parameters to test
    kernels = ['linear', 'rbf', 'poly']
    C_values = [0.1, 1.0, 10.0]
    
    for kernel in kernels:
        for C in C_values:
            predictions, svm_clf = svm_classifier(X_train, y_train, X_test, kernel=kernel, C=C)
            accuracy = calculate_accuracy(y_test, predictions)
    
            print(f"Kernel: {kernel:6} | C: {C:4} | Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_predictions = predictions
                best_params = f"Kernel: {kernel}, C: {C}"
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print_confusion_matrix(y_test, best_predictions)
    print()
def main():
    datasets = [
        ('data/iris/iris.trn', 'data/iris/iris.tst', 'Iris'),
        ('data/optics/opt.trn', 'data/optics/opt.tst', 'Optics'),
        ('data/letter/let.trn', 'data/letter/let.tst', 'Letter'),
        ('data/leukemia/ALLAML.trn', 'data/leukemia/ALLAML.tst', 'Leukemia'),
        ('data/fp/fp.trn', 'data/fp/fp.tst', 'Fp')
    ]
    
    for train_file, test_file, name in datasets:
        evaluate_dataset(train_file, test_file, name)

if __name__ == "__main__":
    main()
