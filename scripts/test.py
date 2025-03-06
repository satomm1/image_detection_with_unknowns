#!/usr/bin/env python3

import rospy
# from utils_package.utils import wrapToPi
import mobileclip
import numpy as np
import torch

if __name__ == "__main__":
    

    rospy.init_node('test')

    # Get the rosparameter clip_model
    clip_model = rospy.get_param('~clip_model')
    print("clip_model:", clip_model)

    root_dir = rospy.get_param('~root_dir')
    print("root_dir:", root_dir)

    # Load the model and tokenizer
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=clip_model, root_dir=root_dir)
    # tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

    print("Load of model and tokenizer successful")

    rospy.spin()
