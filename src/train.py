import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_processing import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from mlflow.tracking import MlflowClient
import joblib
import os


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit_Risk_Classification")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.best_f1 = 0
        self.best_run_id = None
        self.best_model_name = None

    def validate_data(self, X, y):
        """Defensive check: Ensure data integrity before training."""
        if X.empty or y.empty:
            raise ValueError("Input data is empty.")
        if len(X) != len(y):
            raise ValueError(f"Shape mismatch: X has {len(X)} rows, y has {len(y)} rows.")
        if 'is_high_risk' in X.columns:
            raise ValueError("Data Leakage Warning: 'is_high_risk' found in features and dropped.")
        logging.info("Data validation passed.")

    def run_training(self, X, y, model_dict):
        self.validate_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        

        for name, (model, params) in model_dict.items():
            with mlflow.start_run(run_name=name) as run:
                logging.info(f"Training {name}...")
                grid = GridSearchCV(model, params, cv=3, scoring='roc_auc')
                grid.fit(X_train, y_train)
                
                preds = grid.best_estimator_.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, preds),
                    "f1": f1_score(y_test, preds),
                    "roc_auc": roc_auc_score(y_test, preds),
                    "Precision": precision_score(y_test, preds),
                    "Recall": recall_score(y_test, preds),
                }
                
                mlflow.log_params(grid.best_params_)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(grid.best_estimator_, "model")

                if metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    self.best_run_id = run.info.run_id
                    self.best_model_name = name
                    self.best_estimator = grid.best_estimator_
                
                logging.info(f"{name} metrics: {metrics}")

    def register_best_model(self):
        """Registers the model with the highest F1 score."""
        if self.best_run_id:
            model_uri = f"runs:/{self.best_run_id}/model"
            mlflow.register_model(model_uri, "CreditRiskModel")
            logging.info(f"Registered best model: {self.best_model_name} (Run ID: {self.best_run_id})")
        else:
            logging.warning("No model found to register.")

if __name__ == '__main__':
    try:
        logging.info("Loading and processing raw data...")
        df = pd.read_csv("data/raw/data.csv") 
        X, y = process_data(df)
        
        id_cols = [ 'is_high_risk','target','TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'CountryCode', 'CurrencyCode', 'TransactionYear']
        X = X.drop(columns=id_cols, errors='ignore')
        X=X.apply(pd.to_numeric, errors='coerce').fillna(0) # - i did this because XGBoost was struggling with the categories

        trainer = ModelTrainer()
        ratio = (len(y) - sum(y)) / sum(y)
        models = {
        "LogisticRegression": (LogisticRegression(class_weight='balanced'), {'C': [0.1, 1]}),
        "DecisionTree": (DecisionTreeClassifier(class_weight='balanced'), {'max_depth': [5, 10]}),
        "RandomForest": (RandomForestClassifier(class_weight='balanced'), {'n_estimators': [100], 'max_depth': [5, 10]}),
        "XGBoost": (XGBClassifier(scale_pos_weight=ratio), {'learning_rate': [0.1], 'n_estimators': [100]})
    }
        trainer.run_training(X, y, models)
        trainer.register_best_model()
        os.makedirs('models', exist_ok=True)
        joblib.dump(trainer.best_model_name, 'models/best_model.pkl')
        print("Model saved to models/best_model.pkl")
    
        importances = pd.Series(trainer.best_estimator.feature_importances_, index=X.columns)
        print(importances.nlargest(10))
        logging.info("Training process completed successfully.")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")