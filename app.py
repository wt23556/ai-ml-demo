import os
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

# 尝试导入深度学习模型
try:
    from src.deep_models import WideDeepModel, DeepFMModel, TF_AVAILABLE
    DEEP_MODELS_AVAILABLE = TF_AVAILABLE
    if DEEP_MODELS_AVAILABLE:
        logging.info("深度学习模型已加载")
    else:
        logging.warning("TensorFlow 未安装，深度学习模型不可用")
except ImportError as e:
    DEEP_MODELS_AVAILABLE = False
    logging.warning(f"深度学习模型导入失败: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-ML-Demo",
    description="AI Classification/Prediction Model Application with Deep Learning",
    version="2.0.0"
)

# 禁用缓存的中间件
@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    # 对静态图片禁用缓存
    if request.url.path.startswith("/static") and request.url.path.endswith((".png", ".jpg", ".jpeg")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")
model_dir = os.path.join(BASE_DIR, "model")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# 传统模型
data_processor = DataProcessor(data_path=os.path.join(BASE_DIR, "data", "dataset.csv"))
model_trainer = ModelTrainer(model_dir=model_dir)
model_evaluator = ModelEvaluator(output_dir=static_dir)

# 深度学习模型
wide_deep_model = None
deepfm_model = None
if DEEP_MODELS_AVAILABLE:
    wide_deep_model = WideDeepModel(model_dir=os.path.join(BASE_DIR, "model"))
    deepfm_model = DeepFMModel(model_dir=os.path.join(BASE_DIR, "model"))

training_result = None
evaluation_result = None


class PredictRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float


class TrainRequest(BaseModel):
    model_type: Optional[str] = "random_forest"
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None
    min_samples_split: Optional[int] = 2
    min_samples_leaf: Optional[int] = 1
    max_features: Optional[str] = "sqrt"
    bootstrap: Optional[bool] = True
    class_weight: Optional[str] = None
    dataset: Optional[str] = "dataset.csv"


class DeepTrainRequest(BaseModel):
    model_type: Optional[str] = "wide_deep"
    epochs: Optional[int] = 20
    batch_size: Optional[int] = 256
    embedding_dim: Optional[int] = 8
    dataset: Optional[str] = "ad_click_train.csv"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "training_result": training_result,
        "evaluation_result": evaluation_result
    })


@app.post("/train")
async def train_model(params: TrainRequest = None):
    """训练传统机器学习模型 (Random Forest)"""
    global training_result, evaluation_result, data_processor
    
    try:
        if params is None:
            params = TrainRequest()
        
        # 根据参数切换数据集
        dataset_file = params.dataset if params.dataset else "dataset.csv"
        data_path = os.path.join(BASE_DIR, "data", dataset_file)
        
        logger.info(f"Starting training pipeline with dataset: {dataset_file}")
        
        # 使用全局数据处理器
        data_processor = DataProcessor(data_path=data_path)
        df = data_processor.load_data()
        X_train, X_test, y_train, y_test = data_processor.preprocess(df)
        
        # 保存 scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        joblib.dump(data_processor.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        training_result = model_trainer.train(
            X_train, y_train,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            max_features=params.max_features,
            bootstrap=params.bootstrap,
            class_weight=params.class_weight
        )
        
        model_trainer.save_model()
        
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


@app.post("/train/deep")
async def train_deep_model(params: DeepTrainRequest = None):
    """训练深度学习模型 (Wide & Deep / DeepFM)"""
    global training_result, evaluation_result
    
    if not DEEP_MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="深度学习模型不可用，请安装 TensorFlow")
    
    try:
        if params is None:
            params = DeepTrainRequest()
        
        dataset_file = params.dataset if params.dataset else "ad_click_train.csv"
        data_path = os.path.join(BASE_DIR, "data", dataset_file)
        
        # 检查训练集和测试集
        train_path = data_path
        test_path = data_path.replace("train.csv", "test.csv")
        if not os.path.exists(test_path):
            test_path = data_path
        
        logger.info(f"Starting deep learning training with dataset: {dataset_file}")
        logger.info(f"Model type: {params.model_type}")
        
        # 加载数据
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path) if os.path.exists(test_path) and test_path != train_path else None
        
        # 分割训练集为训练和验证
        if df_test is None:
            from sklearn.model_selection import train_test_split
            df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)
        
        logger.info(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")
        
        # 选择和训练模型
        if params.model_type == "wide_deep":
            model = wide_deep_model
            result = model.train(df_train, df_val=df_test, 
                               epochs=params.epochs, batch_size=params.batch_size)
            model.save_model("wide_deep_model")
        elif params.model_type == "deepfm":
            model = deepfm_model
            result = model.train(df_train, df_val=df_test,
                               epochs=params.epochs, batch_size=params.batch_size)
            model.save_model("deepfm_model")
        else:
            raise HTTPException(status_code=400, detail=f"未知的模型类型: {params.model_type}")
        
        # 在测试集上评估
        y_pred = model.predict(df_test)
        y_true = df_test['target'].values
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        test_accuracy = accuracy_score(y_true, y_pred)
        test_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        try:
            y_proba = model.predict_proba(df_test)
            test_auc = roc_auc_score(y_true, y_proba)
        except:
            test_auc = 0.0
        
        evaluation_result = {
            "accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "auc": float(test_auc),
            "model_type": params.model_type
        }
        
        training_result = result
        
        return JSONResponse(content={
            "status": "success",
            "message": f"{params.model_type} model trained successfully",
            "training": result,
            "evaluation": evaluation_result
        })
        
    except Exception as e:
        logger.error(f"Deep learning training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictRequest):
    """使用传统模型预测"""
    try:
        if model_trainer.model is None:
            model_trainer.load_model()
        
        # 加载 scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail="Scaler not found. Please train the model first.")
        
        data_processor.scaler = joblib.load(scaler_path)
        
        data = request.dict()
        X = data_processor.transform_input(data)
        
        prediction = model_trainer.predict(X)[0]
        probabilities = model_trainer.predict_proba(X)[0].tolist()
        
        return JSONResponse(content={
            "status": "success",
            "prediction": int(prediction),
            "probabilities": probabilities,
            "input_features": data,
            "model_type": "random_forest"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status")
async def model_status():
    """查看模型状态"""
    models = {
        "random_forest": os.path.exists(os.path.join(BASE_DIR, "model", "random_forest_model.joblib")),
        "wide_deep": os.path.exists(os.path.join(BASE_DIR, "model", "wide_deep_model.keras")),
        "deepfm": os.path.exists(os.path.join(BASE_DIR, "model", "deepfm_model.keras")),
        "deep_learning_available": DEEP_MODELS_AVAILABLE,
        "training_completed": training_result is not None
    }
    
    return JSONResponse(content=models)


@app.get("/datasets")
async def list_datasets():
    """列出可用的数据集"""
    data_dir = os.path.join(BASE_DIR, "data")
    datasets = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            try:
                import pandas as pd
                df = pd.read_csv(file_path, nrows=5)
                datasets.append({
                    "name": file,
                    "columns": list(df.columns),
                    "preview": df.head(3).to_dict(orient='records')
                })
            except Exception as e:
                datasets.append({
                    "name": file,
                    "error": str(e)
                })
    
    return JSONResponse(content={"datasets": datasets})


@app.get("/metrics")
async def get_metrics():
    """获取评估指标"""
    if evaluation_result is None:
        raise HTTPException(status_code=404, detail="No evaluation results available. Train the model first.")
    
    return JSONResponse(content=evaluation_result)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "AI-ML-Demo",
        "version": "2.0.0",
        "deep_learning_available": DEEP_MODELS_AVAILABLE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
