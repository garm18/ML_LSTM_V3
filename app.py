from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Non-aktifkan pretty print untuk response yang lebih ringkas

MODEL_PATH = 'model/lstm-over-time_model.h5'
SCALER_PATH = 'model/scaler.pkl'
PREDICT_PATH = 'data/rssi_predictions_new.csv'

# **Muat Model dan Scaler**
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model dan Scaler berhasil dimuat!")
except Exception as e:
    print(f"⚠️ Gagal memuat model atau scaler: {str(e)}")

# **Endpoint: Membaca dan Mengonversi CSV ke JSON**
@app.route('/load-data', methods=['GET'])
def load_data():
    try:
        if not os.path.exists(PREDICT_PATH):
            return jsonify({'error': 'CSV file not found'}), 404

        df = pd.read_csv(PREDICT_PATH)
        required_columns = ['timestamp', 'actual_rssi', 'predicted_rssi']
        
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Missing required columns'}), 400

        return jsonify({
            "message": "Data loaded successfully",
            "data": df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# **Endpoint: Prediksi Data Baru**
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Terima data dalam format JSON
        data = request.get_json()
        
        # Pastikan JSON tidak kosong
        if not data or 'rssi_values' not in data:
            return jsonify({'error': 'No RSSI values provided'}), 400

        # Ambil nilai RSSI dari JSON
        rssi_values = data['rssi_values']
        
        # Pastikan input berbentuk list
        if not isinstance(rssi_values, list):
            return jsonify({'error': 'RSSI values must be a list of numbers'}), 400

        # Konversi ke array numpy
        rssi_array = np.array(rssi_values).reshape(-1, 1)

        # Normalisasi data
        scaled_data = scaler.transform(rssi_array)

        # Reshape untuk model LSTM (batch_size, sequence_length, features)
        sequence_length = 10
        if len(scaled_data) < sequence_length:
            return jsonify({'error': 'Not enough data points for LSTM model'}), 400

        X_input = np.array([scaled_data[-sequence_length:]])
        
        # Prediksi menggunakan model
        prediction = model.predict(X_input)
        
        # Denormalisasi hasil prediksi
        prediction_original = scaler.inverse_transform(prediction.reshape(-1, 1))

        return jsonify({
            'prediction': prediction_original.flatten().tolist(),
            'scaled_prediction': prediction.flatten().tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# **Endpoint: Mengecek Status API**
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
