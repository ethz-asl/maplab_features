#!/usr/bin/env python2

import numpy as np
import sensor_msgs.point_cloud2 as pc2

class PointCloudUtils:
    def __init__(self, config):
        self.config = config

        # TODO(lbern): Find out what does coefficients do and give them meaningful names.
        self.a = 255.0 / np.log(1.0 - self.config.close_point * self.config.flatness_range + self.config.far_point * self.config.flatness_range);
        self.b = self.config.close_point - 1.0 / self.config.flatness_range;
        self.c = 255.0 / np.log(1.0 - self.config.min_intensity * self.config.flatness_intensity + self.config.max_intensity * self.config.flatness_intensity);
        self.d = self.config.min_intensity - 1.0 / self.config.flatness_intensity;

    def convert_msg_to_array(self, pcl_msg):
        points_list = []
        for data in pc2.read_points(pcl_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    # Based on https://github.com/PRBonn/OverlapNet/blob/master/src/utils/utils.py
    def project_cloud_to_2d(self, cloud, fov_up=3.0, fov_down=-25.0, height=64, width=900):
        fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

        x_points = cloud[:, 0]
        y_points = cloud[:, 1]
        z_points = cloud[:, 2]
        intensity = cloud[:, 3]
        range = np.sqrt(x_points ** 2 + y_points ** 2, z_points ** 2) + 1e-6


        pitch = np.arcsin(np.clip(z_points/range, -1, 1))
        yaw = np.arctan2(y_points, -x_points)

        # Get projections in image coords.
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # Scale to image size using angular resolution.
        proj_x *= width  # in [0.0, W]
        proj_y *= height  # in [0.0, H]

        # Round and clamp for use as index.
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(width - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(height - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        proj_range = np.full((height, width), -1, dtype=np.float32)
        proj_intensity = np.full((height, width), -1, dtype=np.float32)
        inpaint_mask = np.full((height, width), 0, dtype=np.uint8)

        # Range and intensity scaling.
        bad_points_mask = range < self.config.close_point
        good_points_mask = range > self.config.close_point
        range[bad_points_mask] = 0.0
        range[good_points_mask] = self.a * np.log((range[good_points_mask] - self.b) * self.config.flatness_range)

        good_points_mask = intensity > self.config.min_intensity
        intensity[intensity < self.config.close_point] = 0.0
        intensity[good_points_mask] = self.c * np.log((intensity[good_points_mask] - self.d) * self.config.flatness_intensity)

        proj_range[proj_y, proj_x] = range
        proj_intensity[proj_y, proj_x] = intensity
        inpaint_mask[proj_range <= 0] = 255
        return proj_range, proj_intensity, inpaint_mask
