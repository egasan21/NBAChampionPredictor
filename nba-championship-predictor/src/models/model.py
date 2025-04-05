from sklearn.metrics import accuracy_score

class ChampionshipPredictor:
    def __init__(self):
        self.model = None  # This will hold your trained model
        
    def train_model(self, X_train, y_train):
        """Train the model"""
        # Example using LogisticRegression
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(class_weight='balanced')
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make class predictions (0 or 1)"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions"""
        return self.model.predict_proba(X)
    
    def evaluate_model(self, X_test, y_test):
        """Calculate accuracy"""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'coef_'):
            return self.model.coef_[0]
        return None