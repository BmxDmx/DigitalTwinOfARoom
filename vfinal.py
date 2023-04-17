import cv2
import torch
from testsort import *
import numpy as np
import time

# import the YOLOv5 Model
model = torch.hub.load('.', 'custom', source='local', path='runs/train/old/exp13/weights/last.pt', force_reload=True)

# Load the video
cap = cv2.VideoCapture('dataset/vvideo.mp4')

# Create a Tracker object
tracker = Tracker()

# Declaring all the variables
person_x, person_y, = 0, 0
# monitor_limit - how many monitors should be monitored at ones.
monitors, monitor_limit = 0, 3
original_monitor_x, original_monitor_y, original_monitor_h, original_monitor_w = 0, 0, 0, 0
monitor_x, monitor_y, monitor_w, monitor_h = 0, 0, 0, 0
monitor_position = []
outside = 0
capture_taken = 0

# Select the auto detection of monitors feature
auto_detect_monitors = False

# Function that saves the images when an event occurs
def create_report(frame):
    # Get the time and format the time
    current_time = time.localtime()
    time_formated = time.strftime('%H-%M-%S  %d-%m-%Y  ', current_time)
    # Naming the images and saving the iamges using the time to format the name
    name = 'DigitalTwin/Detection-' + time_formated + '.jpg'
    cv2.putText(frame, str('Monitor moved at'), (750, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
    cv2.putText(frame, str(time_formated), (700, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
    cv2.imwrite(name, frame)


while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (1020, 640))
#   Parse each frame into to the model to detect objects
        results = model(frame)
        detections = []
#   Assigning the detections
        for index, row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            cl = str(row['name'])
            confidence = int(row['confidence'] * 100)
            detections.append([x1, y1, x2, y2, cl])
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id, cl = box_id
#   Displaying the detection on each frame 
            if cl == 'person':
                person_x = x
                person_y = y
                person = cv2.rectangle(frame, (x, y), (person_x, person_y), (0, 155, 200), 2)
            cv2.rectangle(frame, (x, y), (w, h), (255, 2, 123), 2)
            cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            cv2.putText(frame, str(cl), (x + 10, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            cv2.putText(frame, str(confidence), (x + 10, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
#   If automatic monitor detection is TRUE, the position of the monitor will be monitored
            if auto_detect_monitors:
                if cl == 'monitor':

                    if monitors <= monitor_limit:
#   Creates a box around the monitor with a 40px icrease in each direction to detected if the monitor is moved or not.
                        original_monitor_x, original_monitor_y, original_monitor_w, original_monitor_h = x - 40, y - 40, w + 40, h + 40
                        monitors = monitors + 1

                    monitor_x, monitor_y, monitor_w, monitor_h = x, y, w, h

                area = cv2.rectangle(frame, (original_monitor_x, original_monitor_y),
                                     (original_monitor_w, original_monitor_h), (0.255, 200), 3)
                result = cv2.pointPolygonTest(np.array(
                    [(original_monitor_x, original_monitor_y), (original_monitor_w, original_monitor_h)], np.int32), (int(person_x), int(person_y)), False)
#   If the monitors is not detected in the detection box it's been tempered with
                if monitor_x >= original_monitor_x and monitor_y >= original_monitor_y and monitor_h <= original_monitor_h and monitor_w <= original_monitor_w:
                    outside = 0
#   Setting the monitor coordinates to 0 so they are not kept in the memory after the monitor is not in the detection box
                    monitor_x, monitor_y, monitor_w, monitor_h = 0, 0, 0, 0
                else:
                    outside = outside + 1

                if outside > 30:
                    if capture_taken <= 9:
                        create_report(frame)
                        capture_taken = capture_taken + 1
                    outside = 0
        time.sleep(0.1)
#   Show each frame of the video
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
