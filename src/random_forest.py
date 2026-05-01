from sklearn.ensemble import RandomForestClassifier
from src.pipe import line

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    # model.fit(X_train, y_train)
    return line(model, X_train, y_train)