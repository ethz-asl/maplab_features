#!/usr/bin/env python
from __future__ import print_function

import os
import errno
import sys
import cv2
import numpy as np

import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from maplab_msgs.msg import Features
from std_msgs.msg import MultiArrayDimension

def open_fifo(file_name, mode):
    try:
        os.mkfifo(file_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return open(file_name, mode)

def read_bytes(file, num_bytes):
    bytes = b''
    num_read = 0
    while num_read < num_bytes:
        bytes += file.read(num_bytes - num_read)
        num_read = len(bytes)
    return bytes

class ImageReceiver:
    def __init__(self, image_topic, descriptor_topic):
        # image subscriber
        self.image_sub = rospy.Subscriber(
                image_topic, Image, self.image_callback, queue_size=20)
        self.descriptor_pub = rospy.Publisher(
                descriptor_topic, Features, queue_size=20)

        # CV bridge for conversion
        self.bridge = CvBridge()

        # Pipe for transferring images
        self.fifo_images = open_fifo('/tmp/maplab_features_images', 'wb')
        self.fifo_descriptors = open_fifo(
            '/tmp/maplab_features_descriptors', 'rb')

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                    image_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        # Transmit image for processing
        cv_success, cv_binary = cv2.imencode('.png', cv_image)
        assert(cv_success)
        cv_binary = cv_binary.tobytes()
        num_bytes = np.array([len(cv_binary)], dtype=np.uint32).tobytes()
        self.fifo_images.write(num_bytes)
        self.fifo_images.write(cv_binary)
        self.fifo_images.flush()

        # Receive number of descriptors and size
        descriptor_header = read_bytes(self.fifo_descriptors, 3*4)
        num_bytes, num_keypoints, descriptor_size = np.frombuffer(
            descriptor_header, dtype=np.uint32)
        descriptor_data = read_bytes(self.fifo_descriptors, num_bytes)
        descriptor_data = np.frombuffer(descriptor_data, dtype=np.float32)

        num_cols = descriptor_size + 4 # x, y, desc, score, scale
        assert(descriptor_data.size == num_keypoints * num_cols)
        descriptor_data = np.reshape(descriptor_data, (num_keypoints, num_cols))

        x = descriptor_data[:, 0]
        y = descriptor_data[:, 1]
        scores = descriptor_data[:, 2]
        scales = descriptor_data[:, 3]
        descriptors = descriptor_data[:, 4:].flatten().view(np.uint8)

        # Prepare Maplab feature message
        feature_msg = Features()
        feature_msg.header.stamp = image_msg.header.stamp
        feature_msg.numKeypointMeasurements = int(num_keypoints)
        feature_msg.keypointMeasurementsX = x.tolist()
        feature_msg.keypointMeasurementsY = y.tolist()
        feature_msg.keypointMeasurementUncertainties = scores.tolist()
        feature_msg.keypointScales = scales.tolist()
        feature_msg.keypointScores = scores.tolist()
        feature_msg.descriptors.data = descriptors.tolist()

        # Descriptor array size
        dim0 = MultiArrayDimension()
        dim0.label = 'desc_count'
        dim0.size = int(num_keypoints)
        dim0.stride = int(descriptors.size)

        dim1 = MultiArrayDimension()
        dim1.label = 'desc_size'
        desc_bytes =  int(descriptors.size / num_keypoints)
        assert(desc_bytes * num_keypoints == descriptors.size)
        dim1.size = desc_bytes
        dim1.stride = desc_bytes

        feature_msg.descriptors.layout.dim = [dim0, dim1]
        feature_msg.descriptors.layout.data_offset = 0

        self.descriptor_pub.publish(feature_msg)

        #for kp in xy:
        #    cv2.circle(cv_image, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
        #cv2.imshow("Image window2", cv_image)
        #cv2.waitKey(3)


def main(args):
    image_receiver = ImageReceiver(
            "/VersaVIS/cam0/image_raw", "/VersaVIS/cam0/features")
    rospy.init_node('maplab_features', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
