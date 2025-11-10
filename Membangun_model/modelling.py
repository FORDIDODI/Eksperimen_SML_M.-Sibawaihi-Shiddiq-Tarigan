import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI MLFLOW
# ============================================================

# Set tracking URI (local)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name
mlflow.set_experiment("Diabetes_Prediction_Experiment")

print("=" * 60)
print("MEMULAI TRAINING MODEL")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nMemuat data preprocessing...")
df = pd.read_csv('diabetes_preprocessing.csv')
print(f"Data berhasil dimuat! Shape: {df.shape}")

# Split features dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================
# 2. TRAINING MODEL DENGAN MLFLOW AUTOLOG
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

# Enable autologging
mlflow.sklearn.autolog()

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_Basic"):
    
    # Log additional parameters
    mlflow.log_param("dataset_name", "diabetes")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("n_features", X.shape[1])
    
    # Create and train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model berhasil dilatih!")

    # Simpan model secara eksplisit ke MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")

    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics (autolog sudah mencatat, tapi kita tampilkan)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    print("\n" + "=" * 60)
    print("HASIL EVALUASI MODEL")
    print("=" * 60)
    print(f"Training Accuracy  : {train_acc:.4f}")
    print(f"Test Accuracy      : {test_acc:.4f}")
    print(f"Test Precision     : {test_precision:.4f}")
    print(f"Test Recall        : {test_recall:.4f}")
    print(f"Test F1-Score      : {test_f1:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    print("\nModel dan metrics berhasil disimpan ke MLflow!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Artifact URI: {mlflow.get_artifact_uri()}")

print("\n" + "=" * 60)
print("TRAINING SELESAI!")
print("=" * 60)
print("\nCara melihat hasil:")
print("1. Buka terminal baru")
print("2. Jalankan: mlflow ui")
print("3. Buka browser: http://127.0.0.1:5000")
print("=" * 60)