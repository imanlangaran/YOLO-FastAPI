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
    get_model,
)
from helper import Helper


app = FastAPI(title="YOLOv8 FastAPI", version="0.0.0")


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def healthCheck():
    return {"status": "ok"}


@app.post("/image_detect_to_json", tags=["Detection"])
def image_detect_to_json(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file)

    model = get_model("detect")
    preds = get_predict_df(img, model)

    result = {}
    result["detected_objects"] = json.loads(
        preds[["name", "conf"]].to_json(orient="records")
    )

    return result


@app.post("/image_detect_to_image", tags=["Detection"])
def image_detect_to_image(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    model = get_model("detect")
    preds = get_predict_df(img, model)

    out_img = draw_boxes(img=img, preds=preds)

    return StreamingResponse(
        content=Helper.get_bytes_from_image(img=out_img), media_type="image/jpeg"
    )


@app.post("/image_segment_to_image", tags=["Segmentation"])
def image_segment_to_image(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    model = get_model("segment")
    out_img = get_segmented_img(file=img, model=model)

    return StreamingResponse(
        content=Helper.get_bytes_from_image(img=out_img), media_type="image/jpeg"
    )


@app.post("/image_class_to_json", tags=["Classification"])
def image_class_to_json(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    model = get_model("class")
    classes = get_class_df(file=img, model=model)

    result = {}
    result["top5"] = json.loads(classes[["name", "conf"]].to_json(orient="records"))

    return result


@app.post("/image_pose_to_image", tags=["Pose"])
def custom_image_pose_to_image(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    model = get_model("pose")
    out_img = get_posed_img(file=img, model=model)

    return StreamingResponse(
        content=Helper.get_bytes_from_image(img=out_img), media_type="image/jpeg"
    )


@app.post("/custom_image_detect_to_json", tags=["Custom Dataset"])
def custom_image_detect_to_json(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file)

    model = get_model("custom")
    preds = get_predict_df(img, model)

    result = {}
    result["detected_objects"] = json.loads(
        preds[["name", "conf"]].to_json(orient="records")
    )

    return result


@app.post("/custom_image_detect_to_image", tags=["Custom Dataset"])
def custom_image_detect_to_image(file: bytes = File(...)):
    img = Helper.get_img_from_bytes(file=file)

    model = get_model("custom")
    preds = get_predict_df(img, model)

    out_img = draw_boxes(img=img, preds=preds, draw_label=False)

    return StreamingResponse(
        content=Helper.get_bytes_from_image(img=out_img), media_type="image/jpeg"
    )