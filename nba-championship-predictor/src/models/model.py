from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class ChampionshipPredictor:
    def __init__(self):
        self.model = None  # This will hold your trained model
        
    def train_model(self, X_train, y_train):
        self.model = LogisticRegression(class_weight='balanced')
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def get_feature_importance(self):
        if hasattr(self.model, 'coef_'):
            return self.model.coef_[0]
        return None