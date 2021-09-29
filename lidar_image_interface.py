#!/usr/bin/env python2
import numpy as np

import cv2
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from maplab_msgs.msg import Features
from point_cloud_utils import *
import matplotlib.pyplot as plt

class LidarReceiver:
    def __init__(self, lidar_topic, image_topic):
        # Subscriber and publisher.
        self.pc_sub = rospy.Subscriber(lidar_topic, PointCloud2, self.pointcloud_callback)
        self.descriptor_pub = rospy.Publisher(
                image_topic, Image, queue_size=20)
        rospy.loginfo('[LidarReceiver] Subscribed to {sub}.'.format(sub=lidar_topic))
        rospy.loginfo('[LidarReceiver] Publishing on {pub}.'.format(pub=image_topic))

    def pointcloud_callback(self, msg):
        rospy.loginfo('[LidarReceiver] Received pointcloud.')
        cloud = convert_msg_to_array(msg)

        # TODO(lbern): introduce configs
        # OS-1
        #fov_up=17.5
        #fov_down=-15.5
        proj_H = 64
        proj_W = 1024

        # OS-0
        fov_up=50.5
        fov_down=-47.5
        proj_H = 128
        proj_W = 1024

        proj_range, proj_intensity = project_cloud_to_2d(cloud, fov_up, fov_down, proj_H, proj_W)
        self.visualize_projection(proj_range, proj_intensity)


    def visualize_projection(self, range_img, intensity_img):
        range_img = cv2.cvtColor(range_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # range_img_c = cv2.applyColorMap(range_img, cv2.COLORMAP_JET)
        cv2.imshow("range", range_img)
        cv2.imshow("intensity", intensity_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('lidar_image_converter', anonymous=True)
    receiver = LidarReceiver('/os_cloud_node/points', '/os_cloud_node/images')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
