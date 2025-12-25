import cv2
import numpy as np

canvas = np.zeros((600, 600, 3), dtype="uint8")
cv2.line(canvas, (0, 0), (250, 250), (255, 0, 0), 1)
cv2.rectangle(canvas, (10, 10), (60, 60), (0, 255, 0), -1)  # Filled rectangle -1 to add fill
cv2.circle(canvas, (300, 300), 50, (0, 0, 255), 2)  # Circle with thickness 2
cv2.putText(canvas, "OpenCV", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Canvas with shapes", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()