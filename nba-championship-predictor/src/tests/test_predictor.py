import unittest
from src.predictor import ChampionshipPredictor

class TestChampionshipPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = ChampionshipPredictor()

    def test_load_data(self):
        data = self.predictor.load_data('data/nba_data.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_train_model(self):
        data = self.predictor.load_data('data/nba_data.csv')
        self.predictor.train_model(data)
        self.assertIsNotNone(self.predictor.model)

    def test_predict_championship(self):
        data = self.predictor.load_data('data/nba_data.csv')
        self.predictor.train_model(data)
        prediction = self.predictor.predict_championship({'team_stats': [1, 2, 3]})
        self.assertIn(prediction, ['Team A', 'Team B', 'Team C'])  # Adjust based on actual team names

if __name__ == '__main__':
    unittest.main()