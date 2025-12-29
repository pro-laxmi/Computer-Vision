from collections import defaultdict, deque
import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture('people.mp4')
model = YOLO('yolov8n.pt')

id_maps = {}
next_id = 1
trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

# Video saving setup
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('people_annotated.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(np.int32).tolist()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_maps:
                id_maps[oid] = next_id
                next_id += 1

            if oid in id_maps:
                uid = id_maps[oid]  # ✅ correct
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'ID: {uid}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(annotated_frame, (cx, cy), 3, (0, 0, 255), -1)

                trail[uid].append((cx, cy))
                for i in range(1, len(trail[uid])):
                    cv2.line(annotated_frame, trail[uid][i - 1],
                            trail[uid][i], (255, 0, 0), 2)


    # Show the frame with bounding boxes and tracked persons
    cv2.imshow('Person Tracking', annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()