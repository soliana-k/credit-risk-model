import logging
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.pydantic_models import CreditRiskInput, PredictionResponse
from typing import List, Optional

mlflow.set_tracking_uri("http://127.0.0.1:5000")  
MODEL_NAME = "CreditRiskModel"
MODEL_VERSION = "latest"   

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = FastAPI(
    title="Credit Risk Assessment API",
    description="ML-powered credit risk prediction using MLflow",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ModelManager:
    model = None
    preprocessor = None

manager = ModelManager()

@app.on_event("startup")
async def load_model():
    global model_info
    try:
       
        logging.info(f"Loading model: {MODEL_NAME} version {MODEL_VERSION}")
        manager.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
        
        
        client = mlflow.MlflowClient()
        model_version = client.get_model_version(MODEL_NAME, 1)
        model_info = {
            "model_name": MODEL_NAME,
            "version": model_version.version,
            "run_id": model_version.run_id
        }
        logging.info("Model loaded successfully from MLflow")
        manager.preprocessor = joblib.load("models/preprocessor.pkl")
        logging.info("Pipeline loaded successfully")
        
    except Exception as e:
        logging.warning(f"MLflow load failed: {e}. Trying local model...")
        try:
            manager.model = joblib.load("models/best_model.pkl")
            manager.preprocessor = joblib.load("models/preprocessor.pkl")
            logging.info(" Loaded local model")
        except Exception as e2:
            logging.error(f"Failed to load any model: {e2}")
            manager.model = None


def get_risk_level(prob: float):
    if prob >= 0.7:
        return "High", "Strongly Reject"
    elif prob >= 0.4:
        return "Medium", "Manual Review Required"
    else:
        return "Low", "Approve"


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": manager.model is not None,
        "model_info": model_info
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CreditRiskInput):
    if manager.model is None or manager.preprocessor is None:
        raise HTTPException(status_code=503, detail="Assets not initialized")

    try:
        data_dict = input_data.model_dump()
        
       
        expected_cols = [
            'Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount', 
            'TransactionCount', 'TransactionVariability', 'TransactionHour', 
            'TransactionDay', 'TransactionMonth', 'TransactionYear', 
            'ProductCategory', 'ChannelId', 'PricingStrategy', 'CurrencyCode', 
            'ProviderId', 'ProductId', 'CustomerId'
        ]
        
        
        for col in expected_cols:
            if col not in data_dict or data_dict.get(col) is None:
                if "Id" in col or "Code" in col or "Category" in col:
                    data_dict[col] = "Unknown"
                else:
                    data_dict[col] = 0.0 

        
        df = pd.DataFrame([data_dict])

        
        processed_data = manager.preprocessor.transform(df)
        model_features = manager.model.feature_names_in_
        processed_data = pd.DataFrame(processed_data, columns=model_features)
    
        processed_data = processed_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        
         
        
        if 'FraudResult' not in processed_data.columns:
            processed_data['FraudResult'] = 0
            
        processed_data = processed_data[model_features]
        processed_data = processed_data.reindex(
    columns=manager.model.feature_names_in_,
    fill_value=0
)
        
        prob = manager.model.predict_proba(processed_data)[0][1]
        risk_level, recommendation = get_risk_level(prob)

        return PredictionResponse(
            probability_of_default=round(float(prob), 4),
            risk_level=risk_level,
            recommendation=recommendation,
            model_used=MODEL_NAME
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/batch")
async def predict_batch(inputs: List[CreditRiskInput]):
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    results = []
    for inp in inputs:
        result = await predict(inp)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)