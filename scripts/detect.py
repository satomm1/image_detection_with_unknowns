import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
import message_filters
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import tf
from tf.transformations import euler_from_quaternion

from ultralytics import YOLO
import numpy as np
import cv2
import time

"""
This script performs object detection using a YOLO model on RGB and depth images from a camera.
It publishes images with bounding boxes and detected unknown objects.
Classes:
    Detector: A class that handles the detection of objects using YOLO and publishes results.
Functions:
    __init__(self): Initializes the Detector class, loads the YOLO model, and sets up publishers and subscribers.
    unifiedCallback(self, rgb_data, depth_data): Callback function that processes synchronized RGB and depth images, performs object detection, and publishes results.
    run(self): Spins the ROS node to keep it running.
Constants:
    KNOWN_OBJECT_THRESHOLD (float): Confidence threshold for known objects.
    UNKNOWN_OBJECT_THRESHOLD (float): Confidence threshold for unknown objects.
    IOU_THRESHOLD (float): Intersection over Union threshold for Non-Maximum Suppression.
    COLORS (list): List of colors for bounding boxes.
    GEMINI_COLORS (list): List of color names for unknown objects.
    COLOR_CODES (list): List of color codes for unknown objects.
Usage:
    Run this script as a ROS node to perform object detection on incoming camera images and publish results.
"""

KNOWN_OBJECT_THRESHOLD = 0.4
UNKNOWN_OBJECT_THRESHOLD = 0.1

IOU_THRESHOLD = 0.1  # Set the IoU threshold for NMS
COLORS = [(255,50,50), (207,49,225), (114,15,191), (22,0,222), (0,177,122), (34,236,169),
          (34,236,81), (203,203,47), (205,90,23), (102,68,16), (168,215,141)]

GEMINI_COLORS = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'yellow']
COLOR_CODES = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128), (255, 192, 203), (255, 165, 0), (255, 255, 0)]

class Detector:
    
    def __init__(self):
        weights_file = rospy.get_param('~weights_file', '../weights/osod.pt')
        self.model = YOLO(weights_file)
        labels_file = rospy.get_param('~labels_file', 'labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = f.read().splitlines()
        print("Using model " + weights_file)

        # FIXME: Load from a parameter
        self.tall = False

        # Create the publisher that will show image with bounding boxes
        self.boxes_publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)

        # Create the publisher that sends unknown objects
        self.unknown_pub = rospy.Publisher('/unknown_objects', DetectedObjectWithImageArray, queue_size=1)

        # Subscribe to the RGB and depth images, and create a time synchronizer
        self.rbg_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rbg_sub, self.depth_sub], 1, 0.1)
        self.ts.registerCallback(self.unifiedCallback)

    def unifiedCallback(self, rgb_data, depth_data):

        start_time = time.time()

        # Get the image from the message
        image = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)

        if self.tall:
            # Rotate image 180 degrees if mounted on tall robot
            image = cv2.rotate(image, cv2.ROTATE_180)

        # Perform object detection using YOLO
        results = self.model.predict(image, device=0, conf=0.20, agnostic_nms=True, iou=IOU_THRESHOLD, verbose=False)

        detect_time = time.time()
        
        # Get the bounding boxes, confidence, and class labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        num_detected = len(cls)

        # Create the DetectedObjectImageArray message for storing unknown objects
        unknown_object_array = DetectedObjectWithImageArray()
        unknown_object_array.header.stamp = rospy.Time.now()
        unknown_object_array.header.frame_id = 'map'
        unknown_object_array.objects = []

        image_with_boxes = image.copy()  # Image to show all bounding boxes
        image_with_unknown_boxes = image.copy()  # Image to show only unknown bounding boxes
        for i in range(num_detected):

            # Make sure the confidence is above the threshold
            if cls[i] == 0 and conf[i] < UNKNOWN_OBJECT_THRESHOLD:
                continue
            elif cls[i] != 0 and conf[i] < KNOWN_OBJECT_THRESHOLD:
                continue

            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Draw box and label
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), COLORS[cls[i]], 2)
            cv2.putText(image_with_boxes, f"{self.labels[cls[i]]} {conf[i]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[cls[i]], 2)

            # If the object is unknown, add it to the unknown_object_array and draw on unknown image
            if cls[i] == 0 and len(unknown_object_array.objects) < len(GEMINI_COLORS):
                unknown_object = DetectedObjectWithImage()
                unknown_object.class_name = "unknown"
                unknown_object.probability = conf[i]
                unknown_object.color = GEMINI_COLORS[len(unknown_object_array.objects)]
                cv2.rectangle(image_with_unknown_boxes, (x1, y1), (x2, y2), COLOR_CODES[len(unknown_object_array.objects)], 2)
                unknown_object_array.objects.append(unknown_object)

        # Publish the image with all bounding boxes
        image_msg = Image()
        image_msg.data = image_with_boxes.tobytes()
        image_msg.height = image_with_boxes.shape[0]
        image_msg.width = image_with_boxes.shape[1]
        image_msg.encoding = 'rgb8'
        image_msg.step = 3 * image_with_boxes.shape[1]
        image_msg.header.stamp = rospy.Time.now()
        self.boxes_publisher.publish(image_msg)

        # Publish the unknown objects
        if len(unknown_object_array.objects) > 0:
            unknown_object_array.header.stamp = rospy.Time.now()
            unknown_object_array.header.frame_id = 'map'

            # Convert image to the ROS format
            _, buffer = cv2.imencode('.jpg', image_with_unknown_boxes)
            unknown_object_array.data = np.array(buffer).tobytes()

            self.unknown_pub.publish(unknown_object_array)  # Publish the unknown objects

        end_time = time.time()
        # print('Detection time: {}'.format(detect_time - start_time))
        # print('Elapsed time: {}'.format(end_time - start_time))

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('object_detection', anonymous=True)

    detector = Detector()
    detector.run()