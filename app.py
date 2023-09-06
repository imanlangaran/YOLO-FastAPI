from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
import pandas as pd
import numpy as np
from PIL import Image


model = YOLO('models/yolov8n.pt')
def get_predict_df(file : Image) -> pd.DataFrame:
    # model = YOLO('models/yolov8n.pt')

    result = model.predict(source=file, conf=0.6)
    bboxes = pd.DataFrame(result[0].to('cpu').numpy().boxes.xyxy, columns=['x1', 'y1', 'x2', 'y2'])
    bboxes['conf'] = result[0].to('cpu').numpy().boxes.conf
    bboxes['class'] = (result[0].to('cpu').numpy().boxes.cls).astype(int)
    bboxes['name'] = bboxes['class'].replace(result[0].names)
    
    return bboxes


def draw_boxes(img : Image, preds:pd.DataFrame) -> Image:
    annotator = Annotator(np.array(img))
    preds = preds.sort_values(by=['x1'], ascending=True)
    for i, data in preds.iterrows():
        annotator.box_label(
            box=[
                data['x1'],data['y1'],
                data['x2'],data['y2']
            ],
            label=f"{data['name']}: {data['conf']:.2f}", 
            color=colors(data['class'], bgr=True))
    return Image.fromarray(annotator.result())