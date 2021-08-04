#!/usr/bin/env python
from __future__ import print_function

import os
import errno
import sys
import rospy
import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageReceiver:
    def __init__(self):
        # image subscriber
        self.image_sub = rospy.Subscriber(
                "/VersaVIS/cam0/image_raw", Image,self.callback, queue_size=20)

        # CV bridge for conversion
        self.bridge = CvBridge()

        # Pipe for transferring images
        self.fifo_file = '/tmp/image_pipe'
        try:
            os.mkfifo(self.fifo_file)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                    data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        ret, cv_binary = cv2.imencode('.png', cv_image)
        with open(self.fifo_file, 'wb') as fifo:
            fifo.write(cv_binary)

def main(args):
    image_receiver = ImageReceiver()
    rospy.init_node('image_receiver', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
