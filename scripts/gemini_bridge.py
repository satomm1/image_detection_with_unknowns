import rospy
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray

import requests
import shutil
import cv2
import numpy as np
import json
"""
This script defines the GeminiBridge class, which acts as a bridge between ROS and a server hosting an LLM API.
The purpose of this bridge is to label unknown objects detected in images by sending the images to the server
and receiving the names of the most prominent objects in each bounding box.
Classes:
    GeminiBridge: A class that subscribes to a topic with unknown objects, sends the images to a server for labeling,
                  and publishes the labeled objects.
Functions:
    __init__(self, server='http://127.0.0.1:5000/gemini'): Initializes the GeminiBridge with the server URL and sets up
                                                          ROS publishers and subscribers.
    object_callback(self, msg): Callback function that processes the received images, sends them to the server for labeling,
                                and publishes the labeled objects.
    run(self): Starts the ROS node and keeps it running.
Usage:
    Run this script as a ROS node to start the GeminiBridge. The node will subscribe to the "/unknown_objects" topic,
    process the received images, send them to the server for labeling, and publish the labeled objects to the
    "/labeled_unknown_objects" topic.
"""

QUERY = """Provide the basic name of the most prominent object in each of the bounding boxes delineated by color. 
           Provide as consise of a name as possible. For example, "dog" instead of "black dog".
           If any of the bounding boxes appears to having nothing of note, do not include it in the response.
           The possible colors are red, green, blue, purple, pink, orange, and yellow.
        """

class GeminiBridge:
    
    def __init__(self, server='http://127.0.0.1:5000/gemini'):
        
        # The server to send the images to (Hosts the LLM API)
        self.server = server

        # Publisher for providing names of unknown objects
        self.labeled_pub = rospy.Publisher("/labeled_unknown_objects", DetectedObjectArray, queue_size=1)

        # Subscribe to detected unknown objects
        self.unknown_sub = rospy.Subscriber("/unknown_objects", DetectedObjectWithImageArray, self.object_callback, queue_size=1)

        self.done = False  # For testing only, only process 1 image --- remove later

        
    def object_callback(self, msg):

        # For testing only, only process 1 image --- remove later
        if self.done:
            return
        
        # Save obj.data as an image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Now save to the file
        cv2.imwrite("unknown_object.jpg", img)
        shutil.copyfile("unknown_object.jpg", f'../../../../../gemini_code/unknown_object.jpg')

        # Send the image to the LLM --- Have to send via a POST request since this version of Python doesn't 
        # support the LLM API
        data = {'query': QUERY, 'query_type': 'image', 'image_name': "unknown_object.jpg"}
        response = requests.post(self.server, json=data)
        response = response.json()['response']
        response = json.loads(response)

        # Store the results in a dictionary by color
        results = {}
        for obj in response:
            results[obj['bounding_box_color']] = obj['object_name']

        # Match the objects to the results from the LLM
        matched_object_array = DetectedObjectArray()
        for obj in msg.objects:
            if obj.color in results:
                matched_object = DetectedObject()
                matched_object.class_name = results[obj.color]
                matched_object.pose = obj.pose
                matched_object.width = obj.width
                matched_object.x1 = obj.x1
                matched_object.x2 = obj.x2
                matched_object.y1 = obj.y1
                matched_object.y2 = obj.y2
                matched_object_array.objects.append(matched_object)

        # Publish the names of the new objects if any exist
        if len(matched_object_array.objects) > 0:
            self.labeled_pub.publish(matched_object_array)                    
            
        # For testing only, only process 1 image --- remove later
        self.done = True

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("gemini_bridge")
    bridge = GeminiBridge()
    bridge.run()