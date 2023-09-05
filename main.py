from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse

app = FastAPI(title="YOLOv8 FastAPI", version="0.0.0")

@app.get('/', include_in_schema=False)
def home():
    return RedirectResponse(url='/docs')



@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def healthCheck():
    return {'status': 'ok'}
