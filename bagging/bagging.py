import pandas as pd
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def bagging_classifier_sklearn(X_train, y_train, X_test, n_estimators=10, max_depth=None, random_state=42):

    base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    bagging_classifier = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    
    bagging_classifier.fit(X_train, y_train)
    predictions = bagging_classifier.predict(X_test)
    
    return predictions, bagging_classifier

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
    
    predictions, classifier = bagging_classifier_sklearn(X_train, y_train, X_test, n_estimators=10, max_depth=5)
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
