import pandas as pd
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def decision_tree_classifier(X_train, y_train, X_test, max_depth=None, random_state=42):
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_classifier.fit(X_train, y_train)
    
    predictions = dt_classifier.predict(X_test)
    
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
    
def evaluate_dataset(train_file, test_file, dataset_name):
    print(f"Dataset: {dataset_name}")
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    max_depths = [5]
    
    best_accuracy = 0
    best_depth = None
    best_predictions = None
    
    for depth in max_depths:
        print(f"\nDecision Tree (max_depth={depth if depth else 'None'})")
        print("-" * 40)
        
        predictions = decision_tree_classifier(X_train, y_train, X_test, max_depth=depth)
        accuracy = calculate_accuracy(y_test, predictions)
        
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth
            best_predictions = predictions
        
    print_confusion_matrix(y_test, best_predictions)

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
