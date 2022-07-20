#!/usr/bin/env python2

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import cv2

class PointCloudUtils:
    def __init__(self, config):
        self.config = config
        self.lidar_calibration = np.genfromtxt(
            self.config.lidar_calibration, delimiter=',').astype(np.int32)

        self.range_log_base = 255.0 / np.log(1.0 -
            self.config.close_point * self.config.flatness_range +
            self.config.far_point * self.config.flatness_range);
        self.range_lower_end = self.config.close_point - 1.0 / self.config.flatness_range;
        self.intensity_log_base = 255.0 / np.log(1.0 -
            self.config.min_intensity * self.config.flatness_intensity +
            self.config.max_intensity * self.config.flatness_intensity);
        self.intensity_lower_end = self.config.min_intensity - 1.0 / self.config.flatness_intensity;

    def convert_msg_to_array(self, pcl_msg):
        points_list = []
        time_offsets = []
        for data in pc2.read_points(pcl_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
            time_offsets.append(data[4])
        points_list = np.array(points_list, dtype=np.float32)
        time_offsets = np.array(time_offsets, dtype=np.uint32)

        return points_list, time_offsets

    # Based on https://github.com/PRBonn/OverlapNet/blob/master/src/utils/utils.py
    def project_cloud_to_2d(self, cloud, time_offsets, height=64, width=1024):
        intensity = cloud[:, 3]
        range_xyz = np.linalg.norm(cloud[:, 0:3], axis=1)
        mask = range_xyz > self.config.close_point

        proj_range = np.full((height, width), -1, dtype=np.float32)
        proj_intensity = np.full((height, width), 0, dtype=np.uint8)
        proj_cloud = np.full((height, width, 3), -1, dtype=np.float32)
        proj_time_offset = np.full((height, width), 0, dtype=np.uint32)

        # Range and intensity scaling.
        range_xyz = self.range_log_base * np.log(
            (range_xyz - self.range_lower_end) * self.config.flatness_range)

        good_points_mask = intensity > self.config.min_intensity
        intensity[~good_points_mask] = 0.0
        intensity[good_points_mask] = self.intensity_log_base * np.log(
            (intensity[good_points_mask] - self.intensity_lower_end) *
            self.config.flatness_intensity)

        for i in range(height):
            for j in range(width):
                index = i * width + j
                if mask[index]:
                    x = self.lidar_calibration[index]
                    if x != -1:
                        proj_range[i, x] = range_xyz[index]
                        proj_intensity[i, x] = np.clip(intensity[index], 0, 255)
                        proj_cloud[i, x] = cloud[index, 0:3]
                        proj_time_offset[i, x] = time_offsets[index]

        # Mask of pixels that need inpainting
        inpaint_mask = np.full((height, width), 0, dtype=np.uint8)
        inpaint_mask[proj_range <= 0] = 255

        return proj_range, proj_intensity, inpaint_mask, proj_cloud, proj_time_offset
