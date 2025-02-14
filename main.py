import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
engine = pyttsx3.init()

model = YOLO(r"C:\Users\Lenovo\Downloads\yolo11-pytorch-default-v1\yolo11l.pt")

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

detected_classes = set() 

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break


    results = model(frame)


    for result in results:
        boxes = result.boxes.xyxy  
        confidences = result.boxes.conf  
        class_ids = result.boxes.cls  
        for i in range(len(boxes)):
            if confidences[i] > 0.5:  
                x1, y1, x2, y2 = boxes[i]  
                class_id = int(class_ids[i])
                label = f'Class: {class_id}, Conf: {confidences[i]:.2f}'

                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                
                if class_id not in detected_classes:
                    detected_classes.add(class_id)
                    engine.say(f"Detected {class_id}")  # You can replace class_id with a class name if you have a mapping
                    engine.runAndWait()

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()