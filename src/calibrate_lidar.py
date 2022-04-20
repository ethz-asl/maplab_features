#!/usr/bin/env python2
import numpy as np

import rospy
from rosbag import Bag
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt

from config import LidarImageConfig

class LidarCalibrator:
    def __init__(self, rosbag, target_path):
        self.config = LidarImageConfig()
        self.config.init_from_config()
        self.rosbag = rosbag
        self.target_path = target_path
        self.skip = 20

    def convert_msg_to_array(self, pcl_msg):
        points_list = []
        for data in pc2.read_points(pcl_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    def calibrate(self):
        height = self.config.projection_height
        width = self.config.projection_width

        occurence = []
        for i in range(height * width):
            occurence.append([])

        counter = 0
        for topic, msg, t in Bag(self.rosbag).read_messages():
            if topic == self.config.in_pointcloud_topic:
                counter += 1
                if counter % self.skip:
                    continue
                print(counter)

                cloud = self.convert_msg_to_array(msg)
                range_xyz = np.linalg.norm(cloud[:, 0:3], axis=1)
                yaw = np.arctan2(cloud[:, 1], -cloud[:, 0])

                # Get projections in image coords.
                proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

                # Scale to image size using angular resolution.
                proj_x *= width  # in [0.0, W]

                # Round and clamp for use as index.
                proj_x = np.round(proj_x)
                proj_x = np.minimum(width - 1, proj_x)
                proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

                mask = range_xyz > self.config.close_point
                for i in range(height):
                    for j in range(width):
                        index = i * width + j
                        if mask[index]:
                            occurence[index].append(proj_x[index])

        calibration = []
        for o in occurence:
            index_count = {}
            for index in o:
                if not index in index_count:
                    index_count[index] = 1
                else:
                    index_count[index] += 1

            most_common = -1
            common_count = 0

            for index in index_count:
                if index_count[index] > common_count:
                    most_common = index
                    common_count = index_count[index]

            calibration.append(most_common)

        calibration = np.array(calibration).astype(np.int32)
        np.savetxt(self.target_path, calibration, delimiter=',', fmt='%d')

if __name__ == '__main__':
    rospy.init_node('lidar_calibrator', anonymous=True)

    rosbag = '/scratch/Office_Mitte_1.bag'
    target_path = '/home/andrei/calibration.csv'
    calibrator = LidarCalibrator(rosbag, target_path)
    calibrator.calibrate()
