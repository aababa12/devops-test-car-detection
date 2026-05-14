import cv2
import json
from ultralytics import YOLO

def load_labels(labels_path):
    with open(labels_path, "r") as f:
        data = json.load(f)
    return set(data.get("frames_with_cars", []))

def main():
    print("Car detector started")

    video_path = "data/input_video.mp4"
    labels_path = "data/labels.json"
    output_path = "data/output/metrics.json"

    ground_truth_frames = load_labels(labels_path)

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video")
        return

    frame_number = 0
    checked_frames = []
    detected_frames = set()
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % 10 != 0:
            continue

        checked_frames.append(frame_number)

        results = model(frame, verbose=False)

        car_found = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id == 2 and confidence >= 0.25:
                    car_found = True
                    detections.append({
                        "frame": frame_number,
                        "class": "car",
                        "confidence": round(confidence, 2)
                    })

        if car_found:
            detected_frames.add(frame_number)
            print(f"Frame {frame_number}: car detected")

    cap.release()

    checked_frames_set = set(checked_frames)

    true_positive = len(detected_frames & ground_truth_frames)
    false_positive = len(detected_frames - ground_truth_frames)
    false_negative = len(ground_truth_frames - detected_frames)
    true_negative = len(
        checked_frames_set - detected_frames - ground_truth_frames
    )

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    accuracy = (true_positive + true_negative) / len(checked_frames_set) if checked_frames_set else 0

    metrics = {
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative
        },
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "accuracy": round(accuracy, 2),
        "checked_frames": checked_frames,
        "ground_truth_frames": sorted(list(ground_truth_frames)),
        "detected_frames": sorted(list(detected_frames)),
        "detections": detections
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Detection finished")
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("Accuracy:", round(accuracy, 2))
    print("Metrics saved to:", output_path)

if __name__ == "__main__":
    main()