import pandas as pd
from collections import Counter
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def naive_bayes_classifier(X_train, y_train, X_test):
    # Create and train the Gaussian Naive Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = nb_classifier.predict(X_test)
    
    return predictions

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def print_confusion_matrix(y_true, y_pred):
    # Get unique classes and create confusion matrix
    classes = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print("\nConfusion Matrix:")
    print("=" * 50)
    
    # Print header
    print("Actual\\Predicted", end="")
    for cls in classes:
        print(f"{str(cls):>8}", end="")
    print()
    
    # Print matrix with row labels
    for i, cls in enumerate(classes):
        print(f"{str(cls):<15}", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>8}", end="")
        print()
    
    print("=" * 50)
    
    # Print classification report for additional metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))

def evaluate_dataset(train_file, test_file, dataset_name):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
        
    # Naive Bayes doesn't use k parameter, so we run it once
    print("Gaussian Naive Bayes")
    
    predictions = naive_bayes_classifier(X_train, y_train, X_test)
    
    accuracy = calculate_accuracy(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print_confusion_matrix(y_test, predictions)

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
