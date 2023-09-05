from ultralytics import YOLO
import pandas as pd
from PIL import Image

model = YOLO('models/yolov8n.pt')
def get_predict_df(file : Image) -> pd.DataFrame:
    # model = YOLO('models/yolov8n.pt')

    result = model.predict(source='bus.jpg', conf=0.6)
    bboxes = pd.DataFrame(result[0].to('cpu').numpy().boxes.xyxy, columns=['x1', 'y1', 'x2', 'y2'])
    bboxes['confs'] = result[0].to('cpu').numpy().boxes.conf
    bboxes['classes'] = (result[0].to('cpu').numpy().boxes.cls).astype(int)
    bboxes['classes'] = bboxes['classes'].replace(result[0].names)
    
    return bboxes