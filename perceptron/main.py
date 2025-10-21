import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'[\s,]+', header=None, engine='python')
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    return X, y

def perceptron(X_train, y_train, X_test, random_state=42):
    clf = Perceptron(random_state=random_state, max_iter=100, tol=1e-3, eta0=0.0001)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

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
        
    best_accuracy = 0
    best_predictions = None
    
    print(f"\nPerceptron")
    print("-" * 40)

    predictions = perceptron(X_train, y_train, X_test)
    accuracy = calculate_accuracy(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
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
