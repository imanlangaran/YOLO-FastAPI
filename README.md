# YOLO FastApi

This is a simple implementation of YOLOv8 using FastAPI which contains the following routes:

+ [`/docs` or `/`](#swagger-ui) : open Swagger UI!
+ [`/healthcheck`](#health-check) : if the server is running with no problem, it returns `{'status': 'ok'}`
+ [Detection](#detection)
  + [`/image_detect_to_json`](#image-detect-to-json) : it returns a JSON containing detected objects followed by their Name and confidence score.
  + [`/image_detect_to_image`](#image-detect-to-image) : returns an annotated image with detected objects and their confidence score.
+ [Segmentation](#segmentation)
  + [`/image_segment_to_image`](#segmentation) : segment ...
+ [Classification](#classification)
  + [`/image_class_to_json`](#classification) : classification ...
+ [Pose](#pose)
  + [`/image_pose_to_image`](#pose) : detect pose of human ...

## Swagger UI

![SwaggerUI](/pics/SwaggerUI.png)
[Top](#yolo-fastapi)

## Health check

![healthcheck](/pics/healthcheck.png)
[Top](#yolo-fastapi)

## Detection

### Image detect to json

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

### Image class to json

![image_class_to_json](/pics/image_class_to_json.png)
[Top](#yolo-fastapi)

## Pose

### Image pose to image

![image_pose_to_image](/pics/image_pose_to_image.png)
[Top](#yolo-fastapi)