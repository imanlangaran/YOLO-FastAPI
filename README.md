# YOLO FastApi

This is a simple implementation of YOLOv8 using FastAPI which contains the following routes:

+ [`/docs` or `/`](#swagger-ui) : open Swagger UI!
+ [`/healthcheck`](#health-check) : If the server is running with no problem, it returns `{'status': 'ok'}`.
+ [Detection](#detection)
  + [`/image_detect_to_json`](#image-detect-to-json) : it returns a JSON containing detected objects followed by their Name and confidence score.
  + [`/image_detect_to_image`](#image-detect-to-image) : returns an annotated image with detected objects and their confidence score.
+ [Segmentation](#segmentation)
  + [`/image_segment_to_image`](#segmentation) : requires an image as input and returns a segmented image.
+ [Classification](#classification)
  + [`/image_class_to_json`](#classification) : It returns a JSON format of classified objects along with their confidence score. [^1]
+ [Pose](#pose)
  + [`/image_pose_to_image`](#pose) : it requires an image and returns an annotated image.
+ [Custom Dataset](#custom-dataset)
  + [`/custom_image_detect_to_json`](#custom-dataset) : ---
  + [`/custom_image_detect_to_image`](#custom-image-detect-to-image) : ---

[^1]: note that YOLOv8 [Classify Models](https://docs.ultralytics.com/tasks/classify/#models) are trained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) Dataset.

## Swagger UI

![SwaggerUI](/pics/SwaggerUI.png)
[Top](#yolo-fastapi)

## Health check

![healthcheck](/pics/healthcheck.png)
[Top](#yolo-fastapi)

## Detection

### Image detect to JSON

![image_detect_to_json](/pics/image_detect_to_json.png)
[Top](#yolo-fastapi)

### Image detect to image

![image_detect_to_image](/pics/image_detect_to_image.png)
[Top](#yolo-fastapi)

## Segmentation

### Image segment to image

![image_segment_to_image](/pics/image_segment_to_image.png)
[Top](#yolo-fastapi)

## Classification

### Image class to JSON

![image_class_to_json](/pics/image_class_to_json.png)
[Top](#yolo-fastapi)

## Pose

### Image pose to image

![image_pose_to_image](/pics/image_pose_to_image.png)
[Top](#yolo-fastapi)

## Custom Dataset

### Custom Image Detect to JSON

![custom_image_detect_to_json](/pics/custom_image_detect_to_json.png)
[Top](#yolo-fastapi)

### Custom Image Detect to Image

![custom_image_detect_to_image](/pics/custom_image_detect_to_image.png)
[Top](#yolo-fastapi)