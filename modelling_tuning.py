import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, log_loss)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI MLFLOW
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Diabetes_Prediction_Tuning")

print("=" * 60)
print("TRAINING MODEL DENGAN HYPERPARAMETER TUNING")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nMemuat data preprocessing...")
df = pd.read_csv('diabetes_preprocessing.csv')
print(f"Data berhasil dimuat! Shape: {df.shape}")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================
# 2. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# Parameter grid untuk tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Melakukan Grid Search (ini mungkin memakan waktu)...")

# Grid Search dengan Cross-Validation
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_base, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nGrid Search selesai!")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Model terbaik
best_model = grid_search.best_estimator_

# ============================================================
# 3. MANUAL LOGGING KE MLFLOW
# ============================================================

print("\n" + "=" * 60)
print("MANUAL LOGGING KE MLFLOW")
print("=" * 60)

with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
    
    # ========== LOG PARAMETERS ==========
    print("\nLogging Parameters...")
    
    # Best parameters dari grid search
    mlflow.log_params(grid_search.best_params_)
    
    # Additional parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("dataset_name", "diabetes")
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("feature_names", list(X.columns))
    
    # ========== PREDICTIONS ==========
    print("Melakukan Prediksi...")
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    y_pred_proba_train = best_model.predict_proba(X_train)
    y_pred_proba_test = best_model.predict_proba(X_test)
    
    # ========== CALCULATE METRICS ==========
    print("Menghitung Metrics...")
    
    # Training Metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    train_roc_auc = roc_auc_score(y_train, y_pred_proba_train[:, 1])
    train_log_loss = log_loss(y_train, y_pred_proba_train)
    
    # Test Metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
    test_log_loss = log_loss(y_test, y_pred_proba_test)
    
    # ========== LOG METRICS ==========
    print("Logging Metrics...")
    
    # Training metrics
    mlflow.log_metric("training_accuracy_score", train_accuracy)
    mlflow.log_metric("training_precision_score", train_precision)
    mlflow.log_metric("training_recall_score", train_recall)
    mlflow.log_metric("training_f1_score", train_f1)
    mlflow.log_metric("training_roc_auc", train_roc_auc)
    mlflow.log_metric("training_log_loss", train_log_loss)
    
    # Test metrics (sama dengan autolog)
    mlflow.log_metric("test_accuracy_score", test_accuracy)
    mlflow.log_metric("test_precision_score", test_precision)
    mlflow.log_metric("test_recall_score", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    mlflow.log_metric("test_log_loss", test_log_loss)
    
    # Best CV score
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # ========== LOG MODEL ==========
    print("Logging Model...")
    mlflow.sklearn.log_model(
        best_model, 
        "model",
        input_example=X_train.iloc[:5]
    )
    
    # ========== LOG ARTIFACTS ==========
    print("Logging Artifacts...")
    
    # 1. Feature Importance Plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('feature_importance.png')
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    
    # 3. Classification Report (as text file)
    report = classification_report(y_test, y_pred_test)
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    mlflow.log_artifact('classification_report.txt')
    
    # 4. Model Parameters (as text file)
    with open('model_parameters.txt', 'w') as f:
        f.write("Best Hyperparameters\n")
        f.write("=" * 50 + "\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"{param}: {value}\n")
    mlflow.log_artifact('model_parameters.txt')
    
    # ========== PRINT RESULTS ==========
    print("\n" + "=" * 60)
    print("HASIL EVALUASI MODEL")
    print("=" * 60)
    
    print("\nTraining Metrics:")
    print(f"  Accuracy  : {train_accuracy:.4f}")
    print(f"  Precision : {train_precision:.4f}")
    print(f"  Recall    : {train_recall:.4f}")
    print(f"  F1-Score  : {train_f1:.4f}")
    print(f"  ROC-AUC   : {train_roc_auc:.4f}")
    print(f"  Log Loss  : {train_log_loss:.4f}")
    
    print("\nTest Metrics:")
    print(f"  Accuracy  : {test_accuracy:.4f}")
    print(f"  Precision : {test_precision:.4f}")
    print(f"  Recall    : {test_recall:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")
    print(f"  ROC-AUC   : {test_roc_auc:.4f}")
    print(f"  Log Loss  : {test_log_loss:.4f}")
    
    print(f"\nBest CV Score: {grid_search.best_score_:.4f}")
    
    print("\nModel dan artifacts berhasil disimpan ke MLflow!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Artifact URI: {mlflow.get_artifact_uri()}")

print("\n" + "=" * 60)
print("TRAINING SELESAI!")
print("=" * 60)