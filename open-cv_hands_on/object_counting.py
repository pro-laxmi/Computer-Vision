import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture('bottles.mp4')  # Replace with your video path
model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8 model

unique_ids = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect persons (class ID 0 is for persons)
    results = model.track(frame, classes=[0], persist=True, verbose=False)  # Class 0 is person

    # Check if there are detections
    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(np.int32).tolist()
        # Loop through all detected persons and draw bounding boxes
        for box, oid in zip(results[0].boxes.xywh, ids):  # box is in xywh format
            x, y, w, h = map(int, box)  # Convert to integer
            cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + w // 2), (0, 255, 0), 2)  # Draw box

            # Add the ID label
            cv2.putText(frame, f'ID: {oid}', (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add the count of unique persons
        cv2.putText(frame, f'Unique persons: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add unique person IDs to the set
        unique_ids.update(ids)

    # Show the frame with bounding boxes and tracked persons
    cv2.imshow('Person Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()