import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8n model
cap = cv2.VideoCapture(1)

cv2.imshow('Live Camera Feed', cv2.imread('lol.jpg'))  # Display a sample image window to initialize
