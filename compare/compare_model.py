import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def knn_classifier(X_train, y_train, X_test, n_neighbors=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)
    
    predictions = knn_classifier.predict(X_test)
    
    return predictions

def naive_bayes_classifier(X_train, y_train, X_test):
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    
    predictions = nb_classifier.predict(X_test)
    
    return predictions
def decision_tree_classifier(X_train, y_train, X_test, max_depth=None, random_state=42):
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_classifier.fit(X_train, y_train)
    
    predictions = dt_classifier.predict(X_test)
    
    return predictions

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
    
    return predictions


def random_forest_classifier(X_train, y_train, X_test, max_depth=None, random_state=42):
    rf_classifier = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    rf_classifier.fit(X_train, y_train)
    
    predictions = rf_classifier.predict(X_test)
    
    return predictions

def adaboost_classifier(X_train, y_train, X_test, n_estimators=50, max_depth=1, learning_rate=1.0, random_state=42):
    base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    ada_classifier = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    
    ada_classifier.fit(X_train, y_train)
    predictions = ada_classifier.predict(X_test)
    
    return predictions


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
    
def evaluate_dataset(train_file, test_file, dataset_name):
    print(f"Dataset: {dataset_name}")
    
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    depth = 5
    predictions = random_forest_classifier(X_train, y_train, X_test, max_depth=depth)
    accuracy_random_forest = calculate_accuracy(y_test, predictions)

    predictions = bagging_classifier_sklearn(X_train, y_train, X_test, n_estimators=100, max_depth=depth)
    accuracy_bagging = calculate_accuracy(y_test, predictions)

    predictions = adaboost_classifier(X_train, y_train, X_test, n_estimators=100, max_depth=depth, learning_rate=1.0)
    accuracy_ada_boosting = calculate_accuracy(y_test, predictions)

    predictions = knn_classifier(X_train, y_train, X_test, n_neighbors=5)
    accuracy_knn = calculate_accuracy(y_test, predictions)

    predictions = naive_bayes_classifier(X_train, y_train, X_test)
    accuracy_nb = calculate_accuracy(y_test, predictions)

    predictions = decision_tree_classifier(X_train, y_train, X_test, max_depth=depth)
    accuracy_dt = calculate_accuracy(y_test, predictions)

    print(f"Random Forest Accuracy: {accuracy_random_forest:.4f}")
    print(f"Bagging Accuracy: {accuracy_bagging:.4f}")
    print(f"Ada Boosting Accuracy: {accuracy_ada_boosting:.4f}")
    print(f"KNN Accuracy: {accuracy_knn:.4f}")
    print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
    print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")

    print("-" * 50)

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
