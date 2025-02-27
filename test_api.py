import requests
import json
import pandas as pd
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
        sample_data = {
            "rssi_values": [-10.83, -11.30, -10.33, -11.75, -11.41, -31.80],
            "timestamps": [
                "2025-02-11 05:36:18",
                "2025-02-03 01:26:40",
                "2025-02-17 17:24:42",
                "2025-02-19 15:59:42",
                "2025-02-02 14:17:53",
                "2025-02-06 21:23:59"
            ]
        }
        
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=sample_data
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify structure of response
        self.assertIn('prediction', data)
        self.assertIn('input_length', data)
        self.assertIn('is_anomaly', data)
        self.assertIn('timestamps', data)
        
        # Verify data types
        self.assertIsInstance(data['prediction'], list)
        
        print("Prediction test passed!")
        print(f"Predicted RSSI: {data['prediction']}")
        print(f"Anomalies detected: {sum(data['is_anomaly'])}")
        
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
            json={
                "rssi_values": [-10.83, -11.30, -10.33],
                "timestamps": ["2025-02-11 05:36:18"]
            }
        )
        self.assertEqual(response.status_code, 400)
        
        print("Invalid input tests passed!")
        
    def test_batch_predict(self):
        """Test batch prediction with a sample CSV file"""
        # Create a sample CSV file
        sample_data = pd.DataFrame({
            'timestamp': [
                "2025-02-11 05:36:18",
                "2025-02-03 01:26:40",
                "2025-02-17 17:24:42",
                "2025-02-19 15:59:42",
                "2025-02-02 14:17:53",
                "2025-02-06 21:23:59"
            ],
            'actual_rssi': [
                -10.83128802129999,
                -11.30572313470003,
                -10.33705862409999,
                -11.7504838212,
                -11.4132562584,
                -31.80256386539996
            ]
        })
        
        # Save to a temporary file
        temp_file = "temp_test_rssi.csv"
        sample_data.to_csv(temp_file, index=False)
        
        # Test batch prediction
        with open(temp_file, 'rb') as f:
            response = requests.post(
                f"{self.BASE_URL}/batch-predict",
                files={'file': f},
                data={'rssi_column': 'actual_rssi', 'ts_column': 'timestamp'}
            )
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify structure
        self.assertIn('predictions', data)
        self.assertIn('total_records', data)
        self.assertIn('anomaly_count', data)
        
        print("Batch prediction test passed!")
        print(f"Total records processed: {data['total_records']}")
        print(f"Anomalies detected: {data['anomaly_count']}")
    
    def test_save_predictions(self):
        """Test saving predictions to a file"""
        # Create a sample CSV file
        sample_data = pd.DataFrame({
            'timestamp': [
                "2025-02-11 05:36:18",
                "2025-02-03 01:26:40",
                "2025-02-17 17:24:42",
                "2025-02-19 15:59:42",
                "2025-02-02 14:17:53",
                "2025-02-06 21:23:59"
            ],
            'actual_rssi': [
                -10.83128802129999,
                -11.30572313470003,
                -10.33705862409999,
                -11.7504838212,
                -11.4132562584,
                -31.80256386539996
            ]
        })
        
        # Save to a temporary file
        input_file = "temp_input_rssi.csv"
        output_file = "temp_output_rssi.csv"
        sample_data.to_csv(input_file, index=False)
        
        # Test save predictions
        response = requests.post(
            f"{self.BASE_URL}/save-predictions",
            json={
                'input_file': input_file,
                'output_file': output_file,
                'rssi_column': 'actual_rssi',
                'ts_column': 'timestamp'
            }
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify output file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Read output file
        output_df = pd.read_csv(output_file)
        
        # Verify columns
        expected_columns = ['timestamp', 'actual_rssi', 'predicted_rssi', 'is_anomaly']
        for col in expected_columns:
            self.assertIn(col, output_df.columns)
        
        # Clean up
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)
            
        print("Save predictions test passed!")
        print(f"Output file created with {len(output_df)} records")

def main():
    """Run tests"""
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == '__main__':
    main()