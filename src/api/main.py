import logging
import pandas as pd
import numpy as np
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
        print(f"DEBUG: Type of model is {type(manager.model)}")
        manager.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
        print(f"DEBUG: Type of model is {type(manager.model)}")
        
        
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
        
        df_single = pd.DataFrame([input_data.model_dump()])
        
        dummy_row = df_single.copy()
        dummy_row['CustomerId'] = 'DUMMY_ID' 
        
        df_batch = pd.concat([df_single, dummy_row], axis=0)
        processed_batch = manager.preprocessor.transform(df_batch)

        expected_features = manager.model.feature_names_in_
        if isinstance(processed_batch, np.ndarray):
            processed_data = processed_batch[[0], :]
        else:
            processed_data = processed_batch.iloc[[0]]
        
        processed_data = processed_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        final_df = processed_data.reindex(columns=expected_features, fill_value=0)

        prob = manager.model.predict_proba(final_df)[0][1]
        risk_level, recommendation = get_risk_level(prob)

        return PredictionResponse(
            probability_of_default=round(float(prob), 4),
            risk_level=risk_level,
            recommendation=recommendation,
            model_used=MODEL_NAME
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
       
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")
    


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(inputs: List[CreditRiskInput]):
    if manager.model is None or manager.preprocessor is None:
        raise HTTPException(status_code=503, detail="Assets not initialized")
    
    try:
        
        df_batch = pd.DataFrame([i.model_dump() for i in inputs])
        processed_data = manager.preprocessor.transform(df_batch)
        
        if not isinstance(processed_data, pd.DataFrame):
            processed_data = pd.DataFrame(processed_data)
        
        processed_data = processed_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        final_df = processed_data.reindex(columns=manager.model.feature_names_in_, fill_value=0)
        
        probs = manager.model.predict_proba(final_df)[:, 1]
        
        
        responses = []
        for prob in probs:
            risk_level, recommendation = get_risk_level(prob)
            responses.append(PredictionResponse(
                probability_of_default=round(float(prob), 4),
                risk_level=risk_level,
                recommendation=recommendation,
                model_used=MODEL_NAME
            ))
        
        return responses

    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)