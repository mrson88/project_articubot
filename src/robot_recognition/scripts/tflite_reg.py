import pickle
# from __future__ import division
import os
import argparse
import cv2
import numpy as np
import sys
import time
import threading
from threading import *
import importlib.util
# import robot_rasberry.aragon1 as ag
# import RPi.GPIO as GPIO
import time
from threading import Thread
import tensorflow as tf


# from robot_rasberry.recognition.nhandienv1 import face_cascade, recognizer


def cal_square(numbers):
    print("calculate square number")
    for n in numbers:
        time.sleep(5)
        print('square:', n * n)


arr = [2, 3, 7, 9]
tn = threading.Thread(target=cal_square, args=(arr,))
labels = {"person_name", 0}
with open("/home/mrson/ros2_ws/src/articubot_recognition/scripts/labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}



# do khoang cach vat the
# def dokc():
#     try:
#         global distance
#         GPIO.setmode(GPIO.BCM)
#         # Khởi tạo 2 biến chứa GPIO ta sử dụng
#         GPIO_TRIGGER = 23
#         GPIO_ECHO = 24
#         # Thiết lập GPIO nào để gửi tiến hiệu và nhận tín hiệu
#         GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  # Trigger
#         GPIO.setup(GPIO_ECHO, GPIO.IN)  # Echo
#         # Khai báo này ám chỉ việc hiện tại không gửi tín hiệu điện
#         # qua GPIO này, kiểu kiểu ngắt điện ấy
#         GPIO.output(GPIO_TRIGGER, False)
#         # Cái này mình cũng không rõ, nhưng họ bảo là để khởi động cảm biến
#         time.sleep(0.1)
#         # Kích hoạt cảm biến bằng cách ta nháy cho nó tí điện rồi ngắt đi luôn.
#         GPIO.output(GPIO_TRIGGER, True)
#         time.sleep(0.00001)
#         GPIO.output(GPIO_TRIGGER, False)
#
#         # Đánh dấu thời điểm bắt đầu
#         start = time.time()
#         while GPIO.input(GPIO_ECHO) == 0:
#             start = time.time()
#         # Bắt thời điểm nhận được tín hiệu từ Echo
#         while GPIO.input(GPIO_ECHO) == 1:
#             stop = time.time()
#
#         # Thời gian từ lúc gửi tín hiêu
#         elapsed = stop - start
#
#         # Distance pulse travelled in that time is time
#         # multiplied by the speed of sound (cm/s)
#         distance = elapsed * 34000
#         distance = distance / 2
#
#     finally:
#         # Reset GPIO settings
#         GPIO.cleanup()



class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
#                     default='TFLite_model')  # required=True)
# parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
#                     default='detect.tflite')
# parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
#                     default='labelmap.txt')
# parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
#                     default=0.5)
# parser.add_argument('--resolution',
#                     help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
#                     default='640x480')
# parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
#                     action='store_true')





# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
# args = parser.parse_args()
MODEL_NAME = "TFLite_model"
GRAPH_NAME = "detect.tflite'"
LABELMAP_NAME = "labelmap.txt"
min_conf_threshold = float(0.5)
# resW, resH = args.resolution.split('x')
imW, imH = int(640), int(480)
use_TPU = False


# if pkg:
#     pass
# 
#     from tflite_runtime.interpreter import Interpreter
# #     interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT)
# 
#     #
#     if use_TPU:
#         from tflite_runtime.interpreter import load_delegate
# else:
#     from tensorflow.lite.python.interpreter import Interpreter
# 
#     if use_TPU:
#         from tensorflow.lite.python.interpreter import load_delegate
# 
# # If using Edge TPU, assign filename for Edge TPU model
# if use_TPU:
#     # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
#     if (GRAPH_NAME == 'detect.tflite'):
#         GRAPH_NAME = 'edgetpu.tflite'

    # Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
#     interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream

# time.sleep(1)
# dokc()
# ag.begin()
# tn.start()
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
def start(frame1):


    while True:
        # print(distance)

        second = time.strftime('%S')
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        # Grab frame from video stream
        #frame1 = videostream.read()
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # nhan dien mat
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
        # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                xcentre = (xmin + xmax) / 2
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 1)

                # Draw label
                object_name = labels[int(classes[i])]
                confidence=int(scores[i] * 100)# Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, confidence)  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text
            else:
                object_name=''
                xcentre=0
                confidence=0


        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        if (int(second) % 1) == 0:
            pass
            # dokc()
        # Draw framerate in corner of frame
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),
                    2)
        #cv2.putText(frame, 'kc: {0:.2f}'.format(distance), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),
                     #2)  # Draw label distance


        return frame,object_name,confidence,xcentre
#start()
# Clean up
cv2.destroyAllWindows()
videostream.stop()
