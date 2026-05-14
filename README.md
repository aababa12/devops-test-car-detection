# Car Detection Service

## Overview

This project is a DevOps practical test for a car detection service.

The service:
- Reads a video file
- Runs YOLOv8 inference
- Detects cars only
- Compares detections against labels
- Calculates confusion matrix, precision, recall, and accuracy
- Saves metrics to `data/output/metrics.json`
- Runs locally and with Docker Compose
- Includes Jenkins CI pipeline

## Project Structure

```text
detector/
  app.py

data/
  input_video.mp4
  labels.json
  output/
    metrics.json

Dockerfile
docker-compose.yml
Jenkinsfile
requirements.txt
README.md