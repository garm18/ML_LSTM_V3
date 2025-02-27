import requests
import json
import unittest
import os

class TestRSSIPredictionAPI(unittest.TestCase):
    """Test class for RSSI Prediction API"""
    
    # Base URL for API
    BASE_URL = 'http://localhost:5000'
    
    def setUp(self):
        """Setup before each test"""
        # Test if API is running
        try:
            response = requests.get(f"{self.BASE_URL}/health")
            if response.status_code != 200:
                self.skipTest("API server is not running. Please start the Flask application.")
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Please start the Flask application.")
            
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        self.assertTrue(data['scaler_loaded'])
        print("Health check test passed!")
        
    def test_predict_endpoint(self):
        """Test prediction endpoint with sample data"""
        # Sample RSSI values
        sample_data = [
            {"ts": "2025-02-01 00:00:00", "rssi": -10.83},
            {"ts": "2025-02-02 00:00:00", "rssi": -11.30},
            {"ts": "2025-02-03 00:00:00", "rssi": -10.33},
            {"ts": "2025-02-04 00:00:00", "rssi": -11.75},
            {"ts": "2025-02-05 00:00:00", "rssi": -11.41},
            {"ts": "2025-02-06 00:00:00", "rssi": -31.80}
        ]
        
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=sample_data
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify structure of response
        self.assertIn('timestamp', data)
        self.assertIn('actual_rssi', data)
        self.assertIn('predicted_rssi', data)
        
        # Verify data types
        self.assertIsInstance(data['actual_rssi'], list)
        self.assertIsInstance(data['predicted_rssi'], list)
        
        print("Prediction test passed!")
        print(f"Predicted RSSI: {data['predicted_rssi']}")
        
    def test_invalid_input(self):
        """Test prediction endpoint with invalid input"""
        # Empty data
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json={}
        )
        self.assertEqual(response.status_code, 400)
        
        # Invalid data type
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json={"rssi_values": "not-a-list"}
        )
        self.assertEqual(response.status_code, 400)
        
        # Mismatched timestamps
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=[
                {"ts": "2025-02-01 00:00:00", "rssi": -10.83},
                {"ts": "2025-02-02 00:00:00", "rssi": -11.30},
                {"ts": "2025-02-03 00:00:00", "rssi": -10.33}
            ]
        )
        self.assertEqual(response.status_code, 400)
        
        print("Invalid input tests passed!")
    
def main():
    """Run tests"""
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == '__main__':
    main()
