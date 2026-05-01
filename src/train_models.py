import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from src.gaussian import train_gaussian_nb
from src.logistic_regression import train_logistic_regression
from src.random_forest import train_random_forest

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    
    # Store models
    models = {
        "Naive Bayes": train_gaussian_nb(X_train, y_train),
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train)
    }
    
    results = []

    for name, model in models.items():
        preds = model.predict(X_test)
        
        # Handle regression output (convert to binary if needed)
        if name == "Logistic Regression":
            preds = [1 if p >= 0.5 else 0 for p in preds]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True).round(2)
    return models, results_df.round(2)