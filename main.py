from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI(title="YOLOv8 FastAPI", version="0.0.0")

@app.get('/', include_in_schema=False)
def home():
    return RedirectResponse(url='/docs')



@app.get('/healthcheck')
def healthCheck():
    return {'status': 'ok'}
