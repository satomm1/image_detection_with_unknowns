# image_detection_with_unknowns
This is a ROS Noetic package for detecting both known and unknown objects in images. 

### Installation
Clone this repo with 
```
git clone https://github.com/satomm1/image_detection_with_unknowns.git
```

We use [MobileCLIP](https://github.com/satomm1/ml-mobileclip), but this is already included in the repo. To use MobileCLIP with a ROS launch file, some small modifications are made to the original MobileCLIP codebase (for details on the modifications see the subsection below). Install the required packages for MobileCLIP (this assumes that PyTorch is already installed):
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
##### YOLO Weights
Included in this repo are YOLOv11 weights, `osod.pt`, for detecting common office building objects and unknown objects. The class number -> class name map is located in the `scripts/labels.txt` file. These can be updated with custom weights/labels, but the name of the custom files should be updated in parameters in the launch files.  

##### MobileCLIP Weights
We use the default pretrained weight provided by MobileCLIP. To download them, temporarily clone the MobileCLIP repo:
```
clone https://github.com/apple/ml-mobileclip.git
```
Navigate to the `ml-mobileclip` directory and call:
```
source get_pretrained_models.sh
```
This will download a few models to the `/ml-mobileclip/checkpoints` directory. We use the `mobileclip_s0.pt` model, so you can delete the other models. Move the `checkpoints` directory to the `scripts` directory:
```
image_detection_with_unknowns/
└── scripts/
    └── checkpoints/
        └── mobileclip_s0.pt
```
Feel free to delete the `ml-mobileclip` repo at this point.

### Launch Files
We include the `detect.launch` and `detect_with_dist.launch` launch files. These launch the respective detect script with the gemini bridge.

### Custom ROS Messages
This package relies on messages located in the [mattbot_image_detection](https://github.com/satomm1/mattbot_image_detection.git) package. Alternatively, move these messages to this package (and update the `CMakeLists.txt` file).

### LLM Server
We run the LLM server with the [gemini_api_repo](https://github.com/satomm1/gemini_api.git). Instruction for setting this up and running are included in the [repo](https://github.com/satomm1/gemini_api.git).

### Importing mobileclip when using a Launch file
This is not straightforward. Essentially, you can solve this problem by:
1) Place the package into the `src/` directory
2) Create a `setup.py` file:
```
# setup.py
## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
# fetch values from package.xml
setup_args = generate_distutils_setup(
packages=['test_package'],
package_dir={'': 'src'},
)
setup(**setup_args)
```
3) Uncomment the `catkin_python_setup()` line in the `CMakeLists.txt` file and run `catkin_make` (and don't forget to `source devel/setup.bash`)
4) Import the package like you normally would: `import mobileclip`

For more details on the solution, see this [discussion](https://stackoverflow.com/questions/75275684/importing-python-files-functions-from-the-same-directory-in-ros-as-simple-as-it).