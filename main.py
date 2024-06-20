from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import FishFreshnessModel
import uvicorn
import ssl
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

ssl._create_default_https_context = ssl._create_unverified_context

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detection_model_path = os.getenv('DETECTION_MODEL_PATH')
freshness_model_path = os.getenv('FRESHNESS_MODEL_PATH')
classification_model_path = os.getenv('CLASSIFICATION_MODEL_PATH')
combined_model = FishFreshnessModel(detection_model_path, freshness_model_path,classification_model_path)

@app.get('/load-model')
async def index():
    return {"status": "success","message": "API Ready"}

@app.post('/predict')
async def predict(predict:UploadFile = File(...)):
    try: 
        img_predict = await predict.read()
        result = await combined_model.detect_eye_and_classify_freshness(img_predict, 'lautify.appspot.com')
        return {"status": "success","message": "Prediction successfully","data": result}
            
    except Exception as e:
        return {"status": "error","message": e.args}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
   