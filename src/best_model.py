from sklearn.metrics import accuracy_score, f1_score

def select_best_model(best_models, X_test, y_test):
    
    results = []

    for name, model in best_models.items():
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results.append((name, model, acc, f1))

        print(f"{name} → Accuracy: {acc:.2f}, F1: {f1:.2f}")

    # Select best based on F1 (recommended)
    best = sorted(results, key=lambda x: x[3], reverse=True)[0]

    print("\n🏆 Final Best Model:", best[0])

    return best[1], best[0]   # model, name