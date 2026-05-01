from sklearn.linear_model import LogisticRegression
from src.pipe import line

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    
    return line(model,X_train, y_train)