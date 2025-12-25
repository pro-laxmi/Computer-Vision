import cv2

img1 = cv2.imread('lol.jpg')

if img1 is None:
	raise ValueError("Image not found or could not be loaded")

resizing = cv2.resize(img1, (500, 500))
grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img1, 100, 200)
blur = cv2.GaussianBlur(img1, (5, 5), 0)

cv2.imshow('Image', resizing)
cv2.imshow('Grey Image', grey)
cv2.imshow('Edges', edges)
cv2.imshow('Blurred Image', blur)

cv2.waitKey(0)
cv2.destroyAllWindows()