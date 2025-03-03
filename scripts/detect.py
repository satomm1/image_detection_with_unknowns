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

KNOWN_OBJECT_THRESHOLD = 0.4
UNKNOWN_OBJECT_THRESHOLD = 0.1

IOU_THRESHOLD = 0.1  # Set the IoU threshold for NMS
COLORS = [(255,50,50), (207,49,225), (114,15,191), (22,0,222), (0,177,122), (34,236,169),
          (34,236,81), (203,203,47), (205,90,23), (102,68,16), (168,215,141)]

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
        results = self.model.predict(image, device=0, conf=0.20, agnostic_nms=True, iou=IOU_THRESHOLD, verbose=False)

        detect_time = time.time()
        
        # Draw the bounding boxes
        # image_with_boxes = results[0].plot()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        num_detected = len(cls)

        image_with_boxes = image.copy()
        for i in range(num_detected):
            if cls[i] == 0 and conf[i] < UNKNOWN_OBJECT_THRESHOLD:
                continue
            elif cls[i] != 0 and conf[i] < KNOWN_OBJECT_THRESHOLD:
                continue

            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), COLORS[cls[i]], 2)
            cv2.putText(image_with_boxes, f"{self.labels[cls[i]]} {conf[i]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[cls[i]], 2)

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
        # print('Detection time: {}'.format(detect_time - start_time))
        # print('Elapsed time: {}'.format(end_time - start_time))

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('object_detection', anonymous=True)

    detector = Detector()
    detector.run()