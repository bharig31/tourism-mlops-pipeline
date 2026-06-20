"""
Model Building Script with Experimentation Tracking
Loads train/test data from Hugging Face, builds and tunes models,
logs parameters with MLflow, evaluates performance, and registers best model.
"""

from datasets import Dataset, DatasetDict
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from huggingface_hub import login
import numpy as np
import os
import sys

# Get your Hugging Face username from environment variable or command line argument
HF_USERNAME = os.getenv('HF_USERNAME') or (sys.argv[1] if len(sys.argv) > 1 else 'your-username-here')

# Login to Hugging Face if needed
# login(token="your_huggingface_token")

# Set MLflow tracking URI to temp directory for GitHub Actions compatibility
import os
import tempfile
mlflow_tracking_dir = os.path.join(tempfile.gettempdir(), "mlruns")
os.makedirs(mlflow_tracking_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
mlflow.set_experiment("wellness_tourism_package_prediction")

def load_data():
    """Load and prepare train and test datasets from local file."""
    print("Loading dataset from local file...")
    df = pd.read_csv('../data/tourism.csv')
    
    # Data cleaning
    print("Cleaning data...")
    df = df.drop('CustomerID', axis=1)
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df = df.dropna()
    
    # Encode categorical columns
    print("Encoding categorical columns...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]
    
    # Separate features and target
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    
    # Split into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return metrics, y_pred

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train and tune Decision Tree model."""
    print("\nTraining Decision Tree...")
    
    with mlflow.start_run(run_name="DecisionTree"):
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['f1_score']:.4f}")
        
        return best_model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and tune Random Forest model."""
    print("\nTraining Random Forest...")
    
    with mlflow.start_run(run_name="RandomForest"):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        mlflow.log_params(grid_search.best_params_)
        
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['f1_score']:.4f}")
        
        return best_model, metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and tune XGBoost model."""
    print("\nTraining XGBoost...")
    
    with mlflow.start_run(run_name="XGBoost"):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        from xgboost import XGBClassifier
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        mlflow.log_params(grid_search.best_params_)
        
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['f1_score']:.4f}")
        
        return best_model, metrics

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train and tune Gradient Boosting model."""
    print("\nTraining Gradient Boosting...")
    
    with mlflow.start_run(run_name="GradientBoosting"):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        mlflow.log_params(grid_search.best_params_)
        
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['f1_score']:.4f}")
        
        return best_model, metrics

def select_best_model(models, metrics_list):
    """Select the best model based on F1 score."""
    best_idx = np.argmax([m['f1_score'] for m in metrics_list])
    return models[best_idx], metrics_list[best_idx]

def register_model_to_huggingface(model, model_name):
    """Register the best model to Hugging Face."""
    print(f"\nRegistering model to Hugging Face: {model_name}")
    
    # Save model locally
    joblib.dump(model, f"{model_name}.pkl")
    
    # Upload to Hugging Face
    from huggingface_hub import HfApi
    api = HfApi()
    
    # Create a repository if it doesn't exist
    repo_id = f"{HF_USERNAME}/{model_name}"
    try:
        api.create_repo(repo_id=repo_id, repo_type="model")
    except:
        pass  # Repository already exists
    
    # Upload the model file
    api.upload_file(
        path_or_fileobj=f"{model_name}.pkl",
        path_in_repo=f"{model_name}.pkl",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"Model successfully registered to Hugging Face: {repo_id}")

def main():
    """Main training pipeline."""
    
    if HF_USERNAME == 'your-username-here':
        print("ERROR: Please set HF_USERNAME environment variable or pass username as argument")
        print("Usage: python model_training.py your_huggingface_username")
        return
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train different models
    models = []
    metrics_list = []
    
    # Decision Tree
    dt_model, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test)
    models.append(dt_model)
    metrics_list.append(dt_metrics)
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models.append(rf_model)
    metrics_list.append(rf_metrics)
    
    # XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    models.append(xgb_model)
    metrics_list.append(xgb_metrics)
    
    # Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_test, y_test)
    models.append(gb_model)
    metrics_list.append(gb_metrics)
    
    # Select best model
    best_model, best_metrics = select_best_model(models, metrics_list)
    
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for i, (model, metrics) in enumerate(zip(models, metrics_list)):
        model_names = ["Decision Tree", "Random Forest", "XGBoost", "Gradient Boosting"]
        print(f"{model_names[i]}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    print("\n" + "="*50)
    print(f"BEST MODEL: F1 Score = {best_metrics['f1_score']:.4f}")
    print("="*50)
    
    # Register best model to Hugging Face
    register_model_to_huggingface(best_model, "wellness-tourism-predictor")
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()
