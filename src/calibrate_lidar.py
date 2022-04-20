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
        self.skip = 10

    def convert_msg_to_array(self, pcl_msg):
        points_list = []
        for data in pc2.read_points(pcl_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    def calibrate(self):
        # First iterate over all messages to figure out the fov
        fov_up = []
        fov_down = []
        counter = 0
        for topic, msg, t in Bag(self.rosbag).read_messages():
            if topic == self.config.in_pointcloud_topic:
                counter += 1
                if counter % self.skip:
                    continue

                print(counter)
                cloud = self.convert_msg_to_array(msg)
                range_xyz = np.linalg.norm(cloud[:, 0:3], axis=1)

                mask = range_xyz > 0.5
                range_xyz = range_xyz[mask]
                fov = np.arcsin(cloud[mask, 2] / range_xyz)
                fov_up.append(np.max(fov))
                fov_down.append(np.min(fov))

        fov_up = np.mean(fov_up)
        fov_down = np.mean(fov_down)

        assert fov_up >= 0 and fov_down <= 0, "Weirdly angled LiDAR"

        fov_up = abs(fov_up)
        fov_down = abs(fov_down)

        print('fov_up:', fov_up / np.pi * 180.0)
        print('fov_down:', fov_down / np.pi * 180.0)

        fov = abs(fov_down) + abs(fov_up)
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

                mask = range_xyz > self.config.close_point
                x_points = cloud[mask, 0]
                y_points = cloud[mask, 1]
                z_points = cloud[mask, 2]
                range_xyz = range_xyz[mask]
                indices = np.arange(mask.size)[mask]

                pitch = np.arcsin(np.clip(z_points / range_xyz, -1, 1))
                yaw = np.arctan2(y_points, -x_points)

                # Get projections in image coords.
                proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
                proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

                # Scale to image size using angular resolution.
                proj_x *= width  # in [0.0, W]
                proj_y *= height  # in [0.0, H]

                # Round and clamp for use as index.
                proj_x = np.round(proj_x)
                proj_x = np.minimum(width - 1, proj_x)
                proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

                proj_y = np.round(proj_y)
                proj_y = np.minimum(height - 1, proj_y)
                proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

                for i in range(indices.size):
                    occurence[indices[i]].append(proj_x[i] * 10000 + proj_y[i])

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

            if most_common != -1:
                x = most_common // 10000
                y = most_common % 10000
            else:
                x = -1
                y = -1

            calibration.append((x, y))

        calibration = np.array(calibration).astype(np.int32)
        np.savetxt(self.target_path, calibration, delimiter=',')

if __name__ == '__main__':
    rospy.init_node('lidar_calibrator', anonymous=True)

    rosbag = '/scratch/Office_Mitte_1.bag'
    target_path = '/home/andrei/calibration.csv'
    calibrator = LidarCalibrator(rosbag, target_path)
    calibrator.calibrate()
