import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8n model

image = cv2.imread('lol.jpg')  # Read the input image
if image is None:
	raise ValueError("Image not found or could not be loaded")

results = model(image)
annotated_img = results[0].plot()  # Annotate the image with detection results
cv2.imshow('Object Detection', annotated_img)  # Display the annotated image
cv2.waitKey(0)
cv2.destroyAllWindows()