import cv2

# 1. Create a VideoCapture object
# The argument '0' refers to the default camera (built-in webcam). 
# Use '1', '2', etc., for additional cameras.
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# 2. Loop continuously to read frames
while True:
    # Read a frame from the video source
    # 'ret' is a boolean (True/False) indicating success.
    # 'frame' is the actual image (a NumPy array).
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    # Optional: Display a title on the window
    cv2.imshow("Live Camera Feed", frame)

    # 3. Break the loop when a specific key is pressed
    # 'cv2.waitKey(1)' waits for a key event for 1ms.
    # '0xFF == ord(\'q\')' checks if the 'q' key was pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
