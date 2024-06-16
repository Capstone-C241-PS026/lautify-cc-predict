from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import FishFreshnessModel

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the correct paths for the models
detection_model_path = 'gs://lautify.appspot.com/models/BestModel.pt'
classification_model_path = 'gs://lautify.appspot.com/models/FreshnessModel.h5'

# Instantiate the combined model
combined_model = FishFreshnessModel(detection_model_path, classification_model_path)

@app.get('/')
async def index():
    return {"API Ready"}

@app.post('/predict/')
async def predict(predict:UploadFile = File(...)):
    try: 
        img_predict = await predict.read()
        result = await combined_model.detect_eye_and_classify_freshness(img_predict, 'lautify.appspot.com')
        return {"status": "success","message": "Prediction successfully","data": result}
            
    except Exception as e:
        return {"status": "error","message": e.args}
    
   