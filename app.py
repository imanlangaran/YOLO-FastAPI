from ultralytics import YOLO

# from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.utils.plotting import Annotator, colors
import pandas as pd
import numpy as np
from PIL import Image


model = YOLO("models/yolov8n.pt")
seg_model = YOLO("models/yolov8n-seg.pt")
class_model = YOLO("models/yolov8n-cls.pt")
pose_model = YOLO("models/yolov8n-pose.pt")
# custom_model = YOLO("models/yolov8n-custom.pt")

models = {
    "detect": model,
    "segment": seg_model,
    "class": class_model,
    "pose": pose_model,
    # 'custom':
}


def get_model(model_name: str) -> YOLO:
    return models.get(model_name)


def get_predict_df(file: Image, model: YOLO) -> pd.DataFrame:
    # model = YOLO('models/yolov8n.pt')

    result = model.predict(source=file, conf=0.6)
    bboxes = pd.DataFrame(
        result[0].to("cpu").numpy().boxes.xyxy, columns=["x1", "y1", "x2", "y2"]
    )
    bboxes["conf"] = result[0].to("cpu").numpy().boxes.conf
    bboxes["class"] = (result[0].to("cpu").numpy().boxes.cls).astype(int)
    bboxes["name"] = bboxes["class"].replace(result[0].names)

    return bboxes


def draw_boxes(img: Image, preds: pd.DataFrame) -> Image:
    annotator = Annotator(np.array(img))
    preds = preds.sort_values(by=["x1"], ascending=True)
    for i, data in preds.iterrows():
        annotator.box_label(
            box=[data["x1"], data["y1"], data["x2"], data["y2"]],
            label=f"{data['name']}: {data['conf']:.2f}",
            color=colors(data["class"], bgr=True),
        )
    return Image.fromarray(annotator.result())


def get_segmented_img(file: Image, model:YOLO) -> Image:
    result = model.predict(source=file, conf=0.6)[0]
    out_img = Image.fromarray(result.plot()[..., ::-1])
    return out_img


def get_class_df(file: Image, model:YOLO) -> pd.DataFrame:
    result = model.predict(source=file, conf=0.6)
    classes = pd.DataFrame(result[0].to("cpu").numpy().probs.top5, columns=["class"])
    classes["conf"] = result[0].to("cpu").numpy().probs.top5conf
    classes["name"] = classes["class"].replace(result[0].names)
    return classes


def get_posed_img(file: Image, model:YOLO) -> Image:
    result = model.predict(source=file)
    annotator = Annotator(np.array(file))
    for k in reversed(result[0].keypoints.data):
        annotator.kpts(k, result[0].orig_shape, radius=5, kpt_line=True)
    return Image.fromarray(annotator.result())
