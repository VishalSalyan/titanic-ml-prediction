from sklearn.model_selection import GridSearchCV

def tuning_top_models(X_train, y_train, models_with_params):
    """
    models_with_params = {
        "Model Name": (model_object, param_grid)
    }
    """

    best_models = {}

    for name, (model, param_grid) in models_with_params.items():
        print(f"\n🔍 Tuning {name}...")
        print("Param Grid:", param_grid, "\n")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            verbose=2,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        print(f"\n✅ Best Params for {name}: {grid.best_params_}")
        print(f"⭐ Best CV Score: {round(grid.best_score_ * 100, 2)}%")

        best_models[name] = grid.best_estimator_

    return best_models