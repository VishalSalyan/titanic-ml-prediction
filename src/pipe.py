from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def line(model, X_train, y_train):

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', model)
        ])
    pipeline.fit(X_train, y_train)
    return pipeline
    