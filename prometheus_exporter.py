from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import random

total_predictions = Counter('model_predictions_total', 'Total predictions', ['prediction_class'])
total_requests = Counter('model_requests_total', 'Total requests')
total_errors = Counter('model_errors_total', 'Total errors', ['error_type'])
prediction_confidence = Gauge('model_prediction_confidence', 'Prediction confidence')
model_accuracy = Gauge('model_accuracy', 'Model accuracy')
model_precision = Gauge('model_precision', 'Model precision')
model_recall = Gauge('model_recall', 'Model recall')
model_f1_score = Gauge('model_f1_score', 'Model F1 score')
active_connections = Gauge('model_active_connections', 'Active connections')
response_time = Histogram('model_response_time_seconds', 'Response time', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])
memory_usage = Gauge('model_memory_usage_mb', 'Memory usage MB')
cpu_usage = Gauge('model_cpu_usage_percent', 'CPU usage percent')

def simulate():
    duration = random.uniform(0.1, 2.0)
    response_time.observe(duration)
    
    prediction_class = random.choice(['diabetes', 'no_diabetes'])
    total_predictions.labels(prediction_class=prediction_class).inc()
    
    confidence = random.uniform(0.5, 1.0)
    prediction_confidence.set(confidence)
    
    total_requests.inc()
    
    if random.random() < 0.05:
        error_type = random.choice(['timeout', 'validation', 'internal'])
        total_errors.labels(error_type=error_type).inc()

def update_metrics():
    model_accuracy.set(random.uniform(0.75, 0.85))
    model_precision.set(random.uniform(0.70, 0.80))
    model_recall.set(random.uniform(0.72, 0.82))
    model_f1_score.set(random.uniform(0.73, 0.83))
    active_connections.set(random.randint(0, 50))
    memory_usage.set(random.uniform(100, 500))
    cpu_usage.set(random.uniform(10, 80))

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus Exporter Started")
    print("Metrics: http://localhost:8000/metrics")
    
    model_accuracy.set(0.80)
    model_precision.set(0.75)
    model_recall.set(0.77)
    model_f1_score.set(0.76)
    
    counter = 0
    
    try:
        while True:
            for _ in range(random.randint(1, 5)):
                simulate()
                counter += 1
            
            if counter % 30 == 0:
                update_metrics()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping exporter...")