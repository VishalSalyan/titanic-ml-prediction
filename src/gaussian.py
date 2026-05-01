from sklearn.naive_bayes import GaussianNB
from src.pipe import line

def train_gaussian_nb(X_train, y_train):
    # Initialize the classifier
    model = GaussianNB()

    return line(model, X_train, y_train)
