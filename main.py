import torch
import cv2
# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # cv2.imshow('Input', frame)

    results = model(frame)
    results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
    temp = results.pandas().xyxy[0]

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
print('fin')