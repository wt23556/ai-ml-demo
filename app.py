import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import logging
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.deep_models import WideDeepModel, DeepFMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-ML-Demo",
    description="AI Classification/Prediction Model Application",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)
templates.env.cache = None

data_processor = DataProcessor(
    data_path=os.path.join(BASE_DIR, "data", "dataset.csv"),
    model_dir=os.path.join(BASE_DIR, "model")
)
model_trainer = ModelTrainer(model_dir=os.path.join(BASE_DIR, "model"))
model_evaluator = ModelEvaluator(output_dir=static_dir)

training_result = None
evaluation_result = None


class PredictRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float


class TrainRequest(BaseModel):
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None


class DeepTrainRequest(BaseModel):
    model_type: str = "deepfm"
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 1024


class DeepPredictRequest(BaseModel):
    numerical_1: float
    numerical_2: float
    numerical_3: float
    numerical_4: float
    numerical_5: float
    category_1: str
    category_2: str
    category_3: str
    category_4: str


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(templates_dir, "index.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/train")
async def train_model(params: TrainRequest = None):
    global training_result, evaluation_result
    
    try:
        if params is None:
            params = TrainRequest()
        
        logger.info("Starting training pipeline...")
        
        df = data_processor.load_data()
        X_train, X_test, y_train, y_test = data_processor.preprocess(df)
        
        training_result = model_trainer.train(
            X_train, y_train,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth
        )
        
        model_trainer.save_model()
        data_processor.save_scaler()
        
        y_pred = model_trainer.predict(X_test)
        evaluation_result = model_evaluator.evaluate(
            y_test.values, 
            y_pred, 
            training_result.get("training_history")
        )
        
        return JSONResponse(content={
            "status": "success",
            "message": "Model trained and evaluated successfully",
            "training": training_result,
            "evaluation": evaluation_result
        })
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        if model_trainer.model is None:
            model_trainer.load_model()
        
        data = request.dict()
        X = data_processor.transform_input(data)
        
        prediction = model_trainer.predict(X)[0]
        probabilities = model_trainer.predict_proba(X)[0].tolist()
        
        return JSONResponse(content={
            "status": "success",
            "prediction": int(prediction),
            "probabilities": probabilities,
            "input_features": data
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status")
async def model_status():
    model_path = os.path.join(BASE_DIR, "model", "random_forest_model.joblib")
    model_exists = os.path.exists(model_path)
    
    return JSONResponse(content={
        "model_exists": model_exists,
        "model_path": model_path if model_exists else None,
        "training_completed": training_result is not None
    })


@app.get("/metrics")
async def get_metrics():
    if evaluation_result is None:
        raise HTTPException(status_code=404, detail="No evaluation results available. Train the model first.")
    
    return JSONResponse(content=evaluation_result)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI-ML-Demo"}


deep_model_cache = {}


@app.post("/deep/train")
async def train_deep_model(params: DeepTrainRequest = None):
    global deep_model_cache
    
    try:
        if params is None:
            params = DeepTrainRequest()
        
        logger.info(f"Starting {params.model_type} training...")
        logger.info(f"Epochs: {params.epochs}, Batch size: {params.batch_size}")
        
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "large_dataset.csv"))
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
        
        if params.model_type.lower() == "widedeep":
            model = WideDeepModel(model_dir=os.path.join(BASE_DIR, "model"))
        else:
            model = DeepFMModel(model_dir=os.path.join(BASE_DIR, "model"))
            
        result = model.train(
            df_train, 
            df_val,
            epochs=params.epochs,
            batch_size=params.batch_size
        )
        
        model.save_model()
        deep_model_cache[params.model_type] = model
        
        return JSONResponse(content={
            "status": "success",
            "message": f"{params.model_type} trained successfully",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Deep training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deep/predict")
async def predict_deep(request: DeepPredictRequest):
    global deep_model_cache
    
    try:
        model_type = "deepfm"
        
        if model_type not in deep_model_cache:
            model = DeepFMModel(model_dir=os.path.join(BASE_DIR, "model"))
            model.load_model()
            deep_model_cache[model_type] = model
        else:
            model = deep_model_cache[model_type]
        
        df = pd.DataFrame([request.dict()])
        prediction = model.predict(df)[0]
        probability = float(model.predict_proba(df)[0])
        
        return JSONResponse(content={
            "status": "success",
            "prediction": int(prediction),
            "probability": probability,
            "input_features": request.dict()
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Deep model not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Deep prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/info")
async def dataset_info():
    small_path = os.path.join(BASE_DIR, "data", "dataset.csv")
    large_path = os.path.join(BASE_DIR, "data", "large_dataset.csv")
    
    small_df = pd.read_csv(small_path)
    large_df = pd.read_csv(large_path)
    
    return JSONResponse(content={
        "small_dataset": {
            "rows": small_df.shape[0],
            "columns": small_df.shape[1],
            "features": small_df.columns.tolist()
        },
        "large_dataset": {
            "rows": large_df.shape[0],
            "columns": large_df.shape[1],
            "features": large_df.columns.tolist(),
            "categorical_features": ["category_1", "category_2", "category_3", "category_4"],
            "numerical_features": ["numerical_1", "numerical_2", "numerical_3", "numerical_4", "numerical_5"]
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
