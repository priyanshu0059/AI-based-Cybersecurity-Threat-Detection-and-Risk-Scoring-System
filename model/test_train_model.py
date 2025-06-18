import unittest
from model.train_model import preprocess_data, train_model

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            'stime': [1, 2],
            'proto': ['tcp', 'udp'],
            'srcip': ['192.168.1.1', '192.168.1.2'],
            'sport': [12345, 54321],
            'dstip': ['192.168.1.3', '192.168.1.4'],
            'dsport': [80, 443],
            'state': ['ESTABLISHED', 'CLOSED'],
            'dur': [10, 5],
            'sbytes': [1000, 2000],
            'dbytes': [500, 300],
            'label': ['normal', 'attack']
        }

    def test_preprocess_data(self):
        X, y = preprocess_data(self.dataset)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(len(y), 2)

    def test_train_model(self):
        model = train_model(self.dataset)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()