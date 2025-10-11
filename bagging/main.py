import pandas as pd
from collections import Counter
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def bagging_classifier(X_train, y_train, X_test, n_estimators=10, max_depth=None, random_state=42):
    # Create base estimator (Decision Tree)
    base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    # Create Bagging classifier
    bagging_clf = BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=random_state,
        bootstrap=True  # Bootstrap sampling
    )
    
    # Train the bagging classifier
    bagging_clf.fit(X_train, y_train)
    
    # Make predictions
    predictions = bagging_clf.predict(X_test)
    
    return predictions

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
    
    # Print classification report for additional metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))

def evaluate_dataset(train_file, test_file, dataset_name):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    # Test different parameters for Bagging
    n_estimators_list = [5, 10, 25, 50]
    max_depths = [3, 5, 10, None]
    
    best_accuracy = 0
    best_params = None
    best_predictions = None
    
    for n_est in n_estimators_list:
        for depth in max_depths:
            print(f"\nBagging (n_estimators={n_est}, max_depth={depth if depth else 'None'})")
            print("-" * 50)
            
            predictions = bagging_classifier(X_train, y_train, X_test, 
                                           n_estimators=n_est, max_depth=depth)
            accuracy = calculate_accuracy(y_test, predictions)
            
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (n_est, depth)
                best_predictions = predictions
    
    print(f"\n{'='*40}")
    print(f"BEST RESULT: n_estimators={best_params[0]}, max_depth={best_params[1] if best_params[1] else 'None'}")
    print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"{'='*40}")
    
    print_confusion_matrix(y_test, best_predictions)

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
