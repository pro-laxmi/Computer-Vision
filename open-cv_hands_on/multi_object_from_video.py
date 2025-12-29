import cv2
from ultralytics  import YOLO

cap = cv2.VideoCapture('video.mp4')  # Replace 'video.mp4' with your video file path
model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8n model

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, classes=[0])  # Detect only persons
    annotated_frame = results[0].plot()
    cv2.imshow('Multi-object detection from video', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()