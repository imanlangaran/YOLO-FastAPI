from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse
from PIL import Image
import io
import json

from app import get_predict_df
from helper import Helper


app = FastAPI(title="YOLOv8 FastAPI", version="0.0.0")

@app.get('/', include_in_schema=False)
def home():
    return RedirectResponse(url='/docs')



@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def healthCheck():
    return {'status': 'ok'}


@app.post('/image_detect_to_json')
def image_detect_to_json(file : bytes = File(...)):
    img = Helper.get_img_from_bytes(file)
    
    preds = get_predict_df(img)
    
    result={}
    result['detected_objects'] = json.loads(preds[['classes', 'confs']].to_json(orient='records'))
    
    return result
