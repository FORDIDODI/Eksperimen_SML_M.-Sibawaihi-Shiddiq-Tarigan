from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# ============================================================
# LOAD MODEL
# ============================================================

# Ganti dengan run_id atau path model Anda
MODEL_URI = "runs:/<RUN_ID>/model"  # atau "models:/diabetes_model/production"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"Model berhasil dimuat dari: {MODEL_URI}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Gunakan path yang benar, contoh:")
    print("   - runs:/abc123/model")
    print("   - file:///path/to/mlruns/0/abc123/artifacts/model")
    model = None

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['prediction'])
request_duration = Histogram('request_duration_seconds', 'Request duration')
model_score_gauge = Gauge('model_confidence_score', 'Prediction confidence')

# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'Diabetes Prediction API',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            '/predict': 'POST - Make prediction',
            '/health': 'GET - Health check',
            '/metrics': 'GET - Prometheus metrics'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    start_time = time.time()
    
    try:
        # Parse input
        data = request.get_json()
        
        # Validate input
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data['data'])
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Get confidence
        confidence = float(np.max(prediction_proba, axis=1)[0])
        
        # Update metrics
        pred_label = 'diabetes' if prediction[0] == 1 else 'no_diabetes'
        prediction_counter.labels(prediction=pred_label).inc()
        model_score_gauge.set(confidence)
        
        # Calculate duration
        duration = time.time() - start_time
        request_duration.observe(duration)
        
        # Response
        response = {
            'prediction': int(prediction[0]),
            'prediction_label': pred_label,
            'confidence': confidence,
            'probabilities': {
                'no_diabetes': float(prediction_proba[0][0]),
                'diabetes': float(prediction_proba[0][1])
            },
            'duration_ms': round(duration * 1000, 2)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Diabetes Prediction API")
    print("=" * 60)
    print("Endpoints:")
    print("   - http://localhost:5001/")
    print("   - http://localhost:5001/predict")
    print("   - http://localhost:5001/health")
    print("   - http://localhost:5001/metrics")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)