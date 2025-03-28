class ChampionshipPredictor:
    def __init__(self, model=None):
        self.model = model

    def train_model(self, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

    def predict_championship(self, X_test):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        return self.model.predict(X_test)