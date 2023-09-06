# YOLO FastApi

This is a simple implementation of YOLOv8 using FastAPI which contains the following routes:

- [`/docs` or `/`](#swagger-ui) : open Swagger UI!
- [`/healthcheck`](#health-check) : if the server is running with no problem, it returns `{'status': 'ok'}`
- [`/image_detect_to_json`](#image-detect-to-json) : it returns a JSON containing detected objects followed by their Name and confidence score.
- [`/image_detect_to_image`](#image-detect-to-image) : returns an annotated image with detected objects and their confidence score.

## Swagger UI

![SwaggerUI](/pics/SwaggerUI.png)
[Top](#yolo-fastapi)

## Health check

![healthcheck](/pics/healthcheck.png)
[Top](#yolo-fastapi)

## Image detect to json

![image_detect_to_json](/pics/image_detect_to_json.png)
[Top](#yolo-fastapi)

## Image detect to image

![image_detect_to_image](/pics/image_detect_to_image.png)
[Top](#yolo-fastapi)