import dagshub
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Initialize DagsHub
dagshub.init(repo_owner='username', repo_name='diabetes-mlops', mlflow=True)

# Load data
df = pd.read_csv('diabetes_preprocessing.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with MLflow
with mlflow.start_run(run_name="DagsHub_Advanced"):
    # Parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics - Standard
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # ADVANCED: 2 additional metrics
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    mlflow.log_metric("roc_auc_score", roc_auc)
    mlflow.log_metric("matthews_correlation", mcc)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Model logged to DagsHub!")
    print(f"🔗 View at: https://dagshub.com/FORDIDODI/diabetes-mlops.mlflow")