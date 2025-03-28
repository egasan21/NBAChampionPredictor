import pandas as pd
from models.model import ChampionshipPredictor
from utils.data_processing import load_data, preprocess_data, split_data

class NBAChampionshipPredictor:
    def __init__(self):
        self.model = ChampionshipPredictor()
        self.data = None

    def load_and_prepare_data(self, file_path):
        self.data = load_data(file_path)
        self.data = preprocess_data(self.data)

    def train(self):
        X_train, y_train = split_data(self.data)
        self.model.train_model(X_train, y_train)

    def predict(self, input_data):
        return self.model.predict_championship(input_data)

if __name__ == "__main__":
    predictor = NBAChampionshipPredictor()
    predictor.load_and_prepare_data('data/nba_data.csv')
    predictor.train()
    # Example prediction (input_data should be defined based on the model's requirements)
    # prediction = predictor.predict(input_data)
    # print(prediction)