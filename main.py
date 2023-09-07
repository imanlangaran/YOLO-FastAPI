from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse, StreamingResponse
from PIL import Image
import json

from app import (
    get_predict_df,
    draw_boxes,
    get_segmented_img,
    get_class_df,
    get_posed_img,
)
from helper import Helper


app = FastAPI(title="YOLOv8 FastAPI", version="0.0.0")


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def healthCheck():
    return {"status": "ok"}


@app.post("/image_detect_to_json")
def image_detect_to_json(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file)

    preds = get_predict_df(img)

    result = {}
    result["detected_objects"] = json.loads(
        preds[["name", "conf"]].to_json(orient="records")
    )

    return result


@app.post("/image_detect_to_image")
def image_detect_to_image(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    preds = get_predict_df(img)

    out_img = draw_boxes(img=img, preds=preds)

    return StreamingResponse(
        content=Helper.get_bytes_from_image(img=out_img), media_type="image/jpeg"
    )


@app.post("/image_segment_to_image")
def image_segment_to_image(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    out_img = get_segmented_img(file=img)

    return StreamingResponse(
        content=Helper.get_bytes_from_image(img=out_img), media_type="image/jpeg"
    )


@app.post("/image_class_to_json")
def image_class_to_json(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    classes = get_class_df(file=img)

    result = {}
    result["top5"] = json.loads(classes[["name", "conf"]].to_json(orient="records"))

    return result


@app.post("/image_pose_to_image")
def image_pose_to_image(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    out_img = get_posed_img(file=img)

    return StreamingResponse(
        content=Helper.get_bytes_from_image(img=out_img), media_type="image/jpeg"
    )
