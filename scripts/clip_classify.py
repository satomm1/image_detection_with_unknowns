import rospy
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray

# Requirements for Mobile Clip
import torch
from PIL import Image
import mobileclip

# Other packages
import numpy as np


class ClipClassify:

    def __init__(self, server='http://127.0.0.1:5000/gemini'):

        # The server to send the images to (Hosts the LLM API)
        self.server = server

        # Load the model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='checkpoints/mobileclip_s0.pt', device=self.device)
        self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

        self.object_names = []
        self.text = None
        self.text_features = None

        # Publisher for providing names of unknown objects
        self.labeled_pub = rospy.Publisher("/labeled_unknown_objects", DetectedObjectArray, queue_size=1)

        # Subscribe to detected unknown objects
        self.unknown_sub = rospy.Subscriber("/unknown_objects", DetectedObjectWithImageArray, self.object_callback, queue_size=1)


    def object_callback(self, msg):

        if len(msg.objects) == 0:
            # No unknown objects detected, do nothing
            return
        
        if len(self.object_names == 0):
            # We don't yet have names, directly ask gemini
            matched_object_array = ask_gemini(msg)

            # Publish the names of the new objects if any exist
            if len(matched_object_array.objects) > 0:
                self.labeled_pub.publish(matched_object_array)     

                # Add the names to the object_names list
                names_changed = False
                for obj in matched_object_array.objects:
                    if obj.class_name not in self.object_names:
                        self.object_names.append(obj.class_name)  
                        names_changed = True

                if names_changed:
                    # Update the text features
                    self.update_text_features()
            return

        # We have names, so we first try to classify with CLIP
        for obj in msg.objects:
            # Extract the image
            img_data = np.frombuffer(obj.data, dtype=np.uint8)
            img = img_data.reshape((obj.y2 - obj.y1, obj.x2 - obj.x1, 3))
            img = Image.fromarray(img)

            # Preprocess the image
            img = self.preprocess(img).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                img = img.to(self.device, dtype=torch.float16)
                img_features = self.model.encode_image(img)

                img_features /= img_features.norm(dim=-1, keepdim=True)

                text_scores = (100.0 * img_features @ self.text_features.T)
                text_probs = text_scores.softmax(dim=-1)
        

    
    def ask_gemini(self, msg)
        
        # Save msg.data as an image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Now save to the file
        cv2.imwrite("unknown_object.jpg", img)
        shutil.copyfile("unknown_object.jpg", f'../../../../../gemini_code/unknown_object.jpg')

        # Send the image to the LLM --- Have to send via a POST request since this version of Python doesn't 
        # support the LLM API
        data = {'query': QUERY, 'query_type': 'image', 'image_name': "unknown_object.jpg"}

        try:
            response = requests.post(self.server, json=data)
        except:
            print("Error sending image to the LLM")
            return
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

        return matched_object_array


    def update_text_features(self):
        # Put "a " in front of each object name
        object_names_with_prefix = ["a " + name for name in self.object_names]

        # Get the text
        self.text = self.tokenizer(object_names_with_prefix).to(self.device)

        # Get the text features
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.text_features = self.model.encode_text(self.text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def run(self):
        rospy.spin()


if __name__ == "__main__":

    rospy.init_node('clip_classify')

    clip = ClipClassify()
    clip.run()


