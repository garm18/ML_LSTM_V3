from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib

app = Flask(__name__)

# Fungsi untuk memprediksi dan mendeteksi anomali menggunakan model yang disimpan
def predict_anomalies(data):
    # Konversi timestamp ke nilai numerik
    data['ts'] = pd.to_datetime(data['ts']).astype(np.int64) / 10**9
    
    # Muat scaler yang disimpan
    scaler = joblib.load('scaler.pkl')
    
    # Normalisasi data menggunakan scaler yang sama
    data['rssi'] = scaler.transform(data[['rssi']])
    
    # Pisahkan fitur dan target
    X_new = data[['rssi']].values
    y_new = data['rssi'].values
    
    # Gunakan TimeSeriesGenerator
    seq_length = 10  # Panjang sekuens untuk model LSTM
    batch_size = 32  # Ukuran batch
    test_generator = TimeseriesGenerator(X_new, y_new, length=seq_length, batch_size=batch_size)
    
    # Muat model yang disimpan
    loaded_model = load_model('lstm-over-time_model.h5')
    
    # Prediksi nilai RSSI menggunakan model yang disimpan
    y_pred_new = loaded_model.predict(test_generator)
    
    # Denormalisasi hasil prediksi
    y_pred_new_denorm = scaler.inverse_transform(y_pred_new)
    y_new_denorm = scaler.inverse_transform(y_new[seq_length:].reshape(-1, 1))
    
    # Konversi timestamp kembali ke format asli
    timestamps = pd.to_datetime(data['ts'][seq_length:], unit='s')
    
    # Buat hasil prediksi dalam format JSON
    results = {
        'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'actual_rssi': y_new_denorm.flatten().tolist(),
        'predicted_rssi': y_pred_new_denorm.flatten().tolist()
    }
    
    return results

@app.route('/predict', methods=['POST'])
def predict():
    # Terima data dalam format JSON
    data = request.get_json()
    df = pd.DataFrame(data)
    
    # Panggil fungsi prediksi
    results = predict_anomalies(df)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)