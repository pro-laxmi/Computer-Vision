import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-seg.pt')
cap = cv2.VideoCapture('people.mp4')

while True:
    ret, frame = cap.read()
    results = model.track(frame, classes=[0], persist=True, verbose=False)

    for r in results:
        annotated_frame = frame.copy()
        if r.masks and r.boxes and r.boxes.id is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()

            for i, mark in enumerate(masks):
                person_id = int(ids[i])
                x1, y1, x2, y2 = map(int, boxes[i])
                mask_resized = cv2.resize(mark.astype(np.uint8) * 255, (frame.shape[1], frame.shape[0]))
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'ID: {person_id}', (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Segmented People with IDs', annotated_frame)
    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()