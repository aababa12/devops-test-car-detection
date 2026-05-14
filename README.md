# devops-test-car-detection

Python application that downloads a video from AWS S3, runs YOLOv8 car-only
detection on each frame, and prints detection results to the console.

## Configuration

Required environment variables:

- `AWS_REGION`: AWS region for the S3 client, for example `us-east-1`
- `S3_BUCKET`: source S3 bucket name
- `S3_KEY`: source video object key

Optional environment variables:

- `DOWNLOAD_DIR`: local download directory, defaults to `downloads`
- `YOLO_MODEL_WEIGHTS`: YOLOv8 weights path/name, defaults to `yolov8n.pt`
- `CONFIDENCE_THRESHOLD`: detection confidence threshold, defaults to `0.25`
- `FRAME_STRIDE`: process every Nth frame, defaults to `1`

AWS credentials are loaded by boto3 using the standard provider chain, including
environment variables such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_SESSION_TOKEN`.

## Run

```powershell
pip install -r requirements.txt

$env:AWS_REGION = "us-east-1"
$env:S3_BUCKET = "my-video-bucket"
$env:S3_KEY = "videos/input.mp4"

python -m detector.app
```

Example console output:

```text
frame=12 class=car confidence=0.842 bbox=(118.4,72.1,391.7,240.8)
frame=13 cars=0
```
