import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
from collections import Counter
from datetime import datetime



model = YOLO('yolov8n.pt')   # or yolov8n.pt for more speed, 's' for slower


#This is for IMAGE DETECTION
img = cv2.imread('boyyy.jpg') #Enter your image path 

if img is None:
    raise IOError("Cant read the image - check the filename/path")

results = model(img, conf=0.5)
annotated_img = results[0].plot()

cv2.imshow('Image Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Video and Camera 
cap = cv2.VideoCapture(0) #if u want to detect video so put the video path in ""
if not cap.isOpened():
    raise IOError("Cant open the camera")

prev_time = 0
detection_counter = Counter() # tracks total counts across the whole session 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame, conf=0.5, imgsz=320, verbose=False)
    annotated_frame = results[0].plot(conf=True) # conf=True is default, so confidence % shows again 

    # Count objects detected in THIS frame 
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        detection_counter[class_name] += 1

        # FPS calculationS 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    cv2.imshow('Object detection by Altamash', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #This is to close detection, Enter 'q' to close detection 
        break


cap.release()
cv2.destroyAllWindows()

# ---- Generate report after the session ends ---- 
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
report_filename = f'detection_report_{timestamp}.txt'

with open(report_filename, 'w') as f:
    f.write(f'Object Detection Report\n')
    f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write('-' * 30 + '\n')
    f.write(f'Total unique object types detected: {len(detection_counter)}\n\n')
    f.write('Detections (frame-count basis):\n')
    for obj, count in detection_counter.most_common():
        f.write(f'  {obj}: {count}\n')

print(f'Report saved as {report_filename}')
print(detection_counter)