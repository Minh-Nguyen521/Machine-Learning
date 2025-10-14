import pandas as pd
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def adaboost_classifier(X_train, y_train, X_test, n_estimators=50, learning_rate=1.0, random_state=42):
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    
    ada_classifier = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    
    ada_classifier.fit(X_train, y_train)
    predictions = ada_classifier.predict(X_test)
    
    return predictions, ada_classifier

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def print_confusion_matrix(y_true, y_pred):
    classes = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    print(cm_df)
    
def evaluate_dataset(train_file, test_file, dataset_name):
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    predictions, classifier = adaboost_classifier(X_train, y_train, X_test, n_estimators=50, learning_rate=1.0)
    accuracy = calculate_accuracy(y_test, predictions)
    
    print(f"{dataset_name}: {accuracy:.4f}")
    print_confusion_matrix(y_test, predictions)

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
