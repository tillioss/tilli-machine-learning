import unittest
import json
from server.server import app


class TestServer(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_odd_days(self):
        response = self.app.post('/predict')
        data = json.loads(response.data)
        self.assertIn('activity', data)
        self.assertIsInstance(data['activity'], str)

    def test_predict_even_days(self):
        response = self.app.post('/predict', json={'emotion': 'happy', 'feedback': 'positive'})
        data = json.loads(response.data)
        self.assertIn('activity', data)
        self.assertIsInstance(data['activity'], str)

    def test_predict_wrong_endpoint(self):
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 405)

    def test_predict_test_endpoint(self):
        response = self.app.get('/predict/test')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'Predict Till Test')

if __name__ == '__main__':
    unittest.main()