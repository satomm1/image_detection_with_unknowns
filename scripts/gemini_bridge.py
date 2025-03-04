import rospy
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray

import requests
import shutil
import cv2
import numpy as np
import json

QUERY = """Provide the basic name of the most prominent object in each of the bounding boxes delineated by color. 
           Provide as consise of a name as possible. For example, "dog" instead of "black dog".
           If any of the bounding boxes appears to having nothing of note, do not include it in the response.
           The possible colors are red, green, blue, purple, pink, orange, and yellow.
        """

class GeminiBridge:
    
    def __init__(self, server='http://127.0.0.1:5000/gemini'):
        
        self.server = server

        # Publisher for the labeled unknown objects
        self.labeled_pub = rospy.Publisher("/labeled_unknown_objects", DetectedObjectArray, queue_size=1)

        # Subscribe to detected objects
        self.unknown_sub = rospy.Subscriber("/unknown_objects", DetectedObjectWithImageArray, self.object_callback, queue_size=1)

        self.done = False

        
    def object_callback(self, msg):

        if self.done:
            return
        
        

        # Save obj.data as an image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Now save to the file
        cv2.imwrite("unknown_object.jpg", img)
        shutil.copyfile("unknown_object.jpg", f'../../../../../gemini_code/unknown_object.jpg')

        data = {'query': QUERY, 'query_type': 'image', 'image_name': "unknown_object.jpg"}
        response = requests.post(self.server, json=data)
        response = response.json()['response']
        response = json.loads(response)

        results = {}
        for obj in response:
            results[obj['bounding_box_color']] = obj['object_name']

        # Match the results with the objects
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



        if len(matched_object_array.objects) > 0:
            self.labeled_pub.publish(matched_object_array)                    
            

        # TODO Publish identified objects
        # if len(object_array.objects) > 0:
        #     self.label_objects(object_array)

        self.done = True

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("gemini_bridge")
    bridge = GeminiBridge()
    bridge.run()