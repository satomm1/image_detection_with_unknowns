# image_detection_with_unknowns
This is a ROS Noetic package for detecting both known and unknown objects in images. 

### Installation
Clone this repo with 
```
git clone https://github.com/satomm1/image_detection_with_unknowns.git
```

We use [MobileCLIP](https://github.com/satomm1/ml-mobileclip), so we also need to clone this repo. You can clone from the original repo or from my forked repo:
```
cd image_detection_with_unknowns/scripts
git clone https://github.com/satomm1/ml-mobileclip.git

# OR 

git clone https://github.com/apple/ml-mobileclip.git
```
Then, we need to install the required packages for MobileCLIP:
```
pip3 install clip-benchmark
pip3 install datasets
pip3 install open-clip-torch
pip3 install timm
```

### Scripts
The package relies on RGB images published to the  `/camera/color/image_raw` topic. If using the (optional) distance mode, the depth pointcloud should be published to the `/camera/depth/image_raw` topic.

- `detect.py`: This script subscribes to the `/camera/color/image_raw` topic, detects both known and unknown objects, publishes the image with bounding boxes to the `/camera/color/image_with_boxes` topic, and publishes the unknown bounding box details to the `/unknown_objects` topic.
- `detect_with_dist.py`: This script subscribes to the `/camera/color/image_raw` and `/camera/depth/image_raw` topics, detects both unknown and unknown objects, and publishes the image with bounding boxes to the `/camera/color/image_with_boxes` topic. Bounding box details for all bounding boxes are published to the `/detected_objects` topic, including their location in the map. Unknown bounding boxes are filtered, so that only unknown objects that don't overlap with existing map obstacles or previously detected objects are sent to the `/unknown_objects` topic (this prevents repetitive calls to the LLM API for the same object).
- `gemini_bridge.py`: This script subscribes to the `/unknown_objects` topic, uses the messages to query the LLM, and publishes results to the `/labeled_unknown_objects` topic.

### Weights
Included in this repo are YOLOv11 weights, `osod.pt`, for detecting common office building objects and unknown objects. The class number -> class name map is located in the `scripts/labels.txt` file. These can be updated with custom weights/labels, but the name of the custom files should be updated in parameters in the launch files.  

### Launch Files
We include the `detect.launch` and `detect_with_dist.launch` launch files. These launch the respective detect script with the gemini bridge.

### Custom ROS Messages
This package relies on messages located in the [mattbot_image_detection](https://github.com/satomm1/mattbot_image_detection.git) package. Alternatively, move these messages to this package (and update the `CMMakeLists.txt` file).

### LLM Server
We run the LLM server with the [gemini_api_repo](https://github.com/satomm1/gemini_api.git). Instruction for setting this up and running are included in the [repo](https://github.com/satomm1/gemini_api.git).

### Importing mobileclip when using a Launch file
This is not straightforward. For more details on a solution, see this [discussion](https://stackoverflow.com/questions/75275684/importing-python-files-functions-from-the-same-directory-in-ros-as-simple-as-it).