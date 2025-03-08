import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from image_detection_with_unknowns.msg import LabeledObject, LabeledObjectArray

import numpy as np
import cv2
import time
import os

class ImageSaver:
    
    def __init__(self):

        # Directory for saving images
        self.save_dir = rospy.get_param("~save_dir", "../images/")
        self.image_dir = os.path.join(self.save_dir, "images")
        self.label_dir = os.path.join(self.save_dir, "labels")
        self.image_count = 0

        # Create directories if they do not exist
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

        # Get the images in the directory
        self.images = os.listdir(self.image_dir)
        self.images.sort()
        self.image_count = len(self.images)

        # Subscriber to detected objects with images
        self.subscriber = rospy.Subscriber("/labeled_unknown_objects", LabeledObjectArray, self.callback, queue_size=3)


    def callback(self, msg):

        if len(msg.objects) == 0:
            # No objects detected, do nothing
            return

        # Get the image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # For each detected object with image
        for obj in msg.objects:
            if obj.class_name == "unknown":
                continue

            x1 = int(obj.x1)
            y1 = int(obj.y1)
            x2 = int(obj.x2)
            y2 = int(obj.y2)

            # Get normalized xywh coordinates
            x = (x1 + x2) / 2 / img.shape[1]
            y = (y1 + y2) / 2 / img.shape[0]
            w = (x2 - x1) / img.shape[1]
            h = (y2 - y1) / img.shape[0]

            # Write the bounding box to the label file
            self.save_label(self.image_count, obj.class_name, x, y, w, h)
            
        # Save the image
        self.save_image(img, self.image_count)
        self.image_count += 1         
            

    def save_image(self, img, count):
        # Save the image to the image directory
        filename = os.path.join(self.image_dir, f"{count:06d}.jpg")
        cv2.imwrite(filename, img)

    def save_label(self, count, name, x, y, w, h):
        # Save the label to the label directory
        filename = os.path.join(self.label_dir, f"{count:06d}.txt")
        with open(filename, "a") as f:
            f.write(f"{name} {x} {y} {w} {h}\n")

    def run(self):  
        # Main loop
        rospy.spin()


if __name__ == "__main__":

    rospy.init_node("image_saver")
    image_saver = ImageSaver()
    image_saver.run()
