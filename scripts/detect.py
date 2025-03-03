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

IOU_THRESHOLD = 0.1  # Set the IoU threshold for NMS
OBJECT_CONFIDENCE_THRESHOLD = 0.7
OTHER_CONFIDENCE_THRESHOLD = 0.03
COLORS = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'yellow']
COLOR_CODES = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128), (255, 192, 203), (255, 165, 0), (255, 255, 0)]


class Detector:
    
    def __init__(self):
        weights_file = rospy.get_param('~weights_file', '../weights/osod.pt')
        self.model = YOLO(weights_file)
        labels_file = rospy.get_param('~coco_labels_file', 'labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = f.read().splitlines()
        print("Using model " + weights_file)

        # FIXME: Load from a parameter
        self.tall = False

        # Create the publisher that will show image with bounding boxes
        self.boxes_publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)

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

        # Perform object detection
        results = self.model.predict(image, device=0, conf=0.03, agnostic_nms=True, iou=IOU_THRESHOLD, verbose=False)

        detect_time = time.time()
        
        # Draw the bounding boxes
        image_with_boxes = results[0].plot()
        num_detected = len(results[0].boxes.cls)

        # Create the ROS Image and publish it
        image_msg = Image()
        image_msg.data = image_with_boxes.tobytes()
        image_msg.height = image_with_boxes.shape[0]
        image_msg.width = image_with_boxes.shape[1]
        image_msg.encoding = 'rgb8'
        image_msg.step = 3 * image_with_boxes.shape[1]
        image_msg.header.stamp = rospy.Time.now()
        self.boxes_publisher.publish(image_msg)

        end_time = time.time()
        print('Detection time: {}'.format(detect_time - start_time))
        print('Elapsed time: {}'.format(end_time - start_time))