import cv2 as cv
import numpy as np
import avisengine
import car_functions_v2 as cf
import onnxruntime as ort
import time

# Creating an instance of the Car class
car = avisengine.Car()

# Connecting to the server
if not car.connect('127.0.0.1', 25001):  # Change to 2500 if that's the correct port
    print("❌ Connection failed. Check if AvisEngine is running on 127.0.0.1:25001.")
    exit()
print("Connected to AvisEngine")

# Load YOLOv11 ONNX model
model = ort.InferenceSession("best.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Preprocess image for YOLO
def preprocess_image(image):
    img = cv.resize(image, (640, 640)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img

# Postprocess YOLO output
def postprocess_output(outputs, orig_w, orig_h, conf_thres=0.4, iou_thres=0.7):
    boxes, scores, class_ids = [], [], []
    out0 = outputs[0][0].T
    classes = ["mane"]  # Only mane class
    for row in out0:
        prob = row[4:].max()
        if prob < conf_thres:
            continue
        xc, yc, w, h = row[:4]
        class_id = row[4:].argmax()
        x1, y1 = (xc - w / 2) / 640 * orig_w, (yc - h / 2) / 640 * orig_h
        x2, y2 = (xc + w / 2) / 640 * orig_w, (yc + h / 2) / 640 * orig_h
        boxes.append([x1, y1, x2, y2])
        scores.append(prob)
        class_ids.append(class_id)
    indices = cv.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    result = [[boxes[i], classes[class_ids[i]], scores[i]] for i in indices]
    return result

# Initial settings
avg = 0
speed = 30
mane_detected = False
start_time = None
frame_count = 0

while True:
    car.getData()

    # Set initial speed
    car.setSpeed(speed)

    # Get sensor data and image
    left_sens, mid_sens, right_sens = car.getSensors()
    frame = car.getImage()

    # Process image for lane detection
    if frame is not None:
        white_img, avg = cf.calc_steering(frame)
        steering = cf.translate(avg, 200, 300, -30, 30)
        car.setSteering(int(steering))

        # Process with YOLO model every 4 frames
        if frame_count % 4 == 0:
            input_image = preprocess_image(frame)
            outputs = model.run(None, {"images": input_image})
            orig_h, orig_w = frame.shape[:2]
            result = postprocess_output(outputs, orig_w, orig_h)
        frame_count = (frame_count + 1) % 4

        # Control logic based on mane detection and path change
        if result:
            label, prob = result[0][1], result[0][2]
            x1, y1, x2, y2 = map(int, result[0][0])
            print(f"Detected: {label}, Confidence: {prob:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")
            
            if label == "mane" and x2 > 200:  # If mane is close
                mane_detected = True
                if start_time is None:
                    start_time = time.time()
                if time.time() - start_time < 2:  # Change path for 2 seconds
                    center_x = (x1 + x2) // 2  # Center of the obstacle
                    if center_x < orig_w // 2:  # Obstacle on the left
                        steering = 45  # Steer right
                    else:  # Obstacle on the right
                        steering = -45  # Steer left
                    car.setSteering(int(steering))
                    speed = 15  # Reduce speed
                else:
                    mane_detected = False
                    start_time = None
            else:
                mane_detected = False
                start_time = None
        else:
            mane_detected = False
            start_time = None

        # If no mane, return to normal speed and steering
        if not mane_detected:
            speed = 30
            car.setSteering(int(steering))

        # Display
        show_frame = frame.copy()
        cv.circle(show_frame, (avg, 300), 3, (255, 0, 0), cv.FILLED)
        for box, label, score in result:
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(show_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(show_frame, f"{label} {score:.2f}", (x1, y1 - 5),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
        show_frame = cv.putText(show_frame, f'current speed: {car.getSpeed()}', (10, 20),
                                cv.FONT_HERSHEY_TRIPLEX, 1, (123, 0, 255), 3)

        cv.imshow('frame', show_frame)
        if cv.waitKey(1) & 0xFF == 27:  # Press Esc to exit
            break

car.stop()
cv.destroyAllWindows()