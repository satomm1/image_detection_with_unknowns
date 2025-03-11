import rospy
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from image_detection_with_unknowns.msg import LabeledObject, LabeledObjectArray
from std_msgs.msg import UInt32
from rospy_message_converter import message_converter
from sensor_msgs.msg import Image

import numpy as np
import os
import cv2
import json
import base64
import time

from cyclonedds.domain import DomainParticipant, DomainParticipantQos
from cyclonedds.topic import Topic
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from cyclonedds.core import Qos, Policy, Listener
from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsParticipant
from dataclasses import dataclass

@dataclass
class DataMessage(IdlStruct):
    message_type: str
    sending_agent: int
    timestamp: int
    data: str


class UnknownObjectSaver:
    """
    This class is responsible for saving unknown objects to a file.
    The unknown objects are received from the unknown_object_detector node.
    They are then identified via an LLM.
    Last, they are saved to a file to be used for training.
    """

    def __init__(self):

        # Get agent id num from environment
        self.agent_id = int(os.environ.get('ROBOT_ID'))
        
        # Reliable qos
        self.reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=10)),
            Policy.Durability.TransientLocal,
            Policy.History.KeepLast(depth=10)
        )

        self.best_effort_qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.Liveliness.ManualByParticipant(lease_duration=duration(milliseconds=30000))
            # Policy.Deadline(duration(milliseconds=1000))
            # Policy.History.KeepLast(depth=1)
        )

        self.lease_duration_ms = 30000
        qos_profile = DomainParticipantQos()
        qos_profile.lease_duration = duration(milliseconds=self.lease_duration_ms)

        # Create a DomainParticipant and Publisher
        self.participant = DomainParticipant()
        self.publisher = Publisher(self.participant)

        # Now publish to 'DataTopic' + agent_id topic
        self.topic = Topic(self.participant, 'DataTopic' + str(self.agent_id), DataMessage)

        # Create a DataWriter
        self.writer = DataWriter(self.publisher, self.topic, qos=self.reliable_qos)

        # Create a subscriber to the labeled_unknown_objects topic
        self.labeled_image_subscriber = rospy.Subscriber("/labeled_unknown_objects", LabeledObjectArray, self.image_callback, queue_size=3)

    def image_callback(self, msg):
        if len(msg.objects) == 0:
            # No objects detected, do nothing
            return

        img_message_str = message_converter.convert_ros_message_to_dictionary(msg)
        message = DataMessage(message_type='unknown_image', sending_agent=self.agent_id, timestamp=rospy.Time.now().to_nsec(), data='')
        message.data = json.dumps(img_message_str) 

        self.writer.write(message)
        print("Sent an image...")
        time.sleep(0.01) 
    
    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("unknown_object_saver")

    unknown_object_saver = UnknownObjectSaver()
    unknown_object_saver.run()
