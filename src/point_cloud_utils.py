#!/usr/bin/env python2

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2

class PointCloudUtils:
    def __init__(self, config):
        self.config = config

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
        for data in pc2.read_points(pcl_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    # Based on https://github.com/PRBonn/OverlapNet/blob/master/src/utils/utils.py
    def project_cloud_to_2d(self, cloud, fov_up=0.0, fov_down=0.0, height=64, width=1024):
        fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

        x_points = cloud[:, 0]
        y_points = cloud[:, 1]
        z_points = cloud[:, 2]
        intensity = cloud[:, 3]
        range_xyz = np.sqrt(x_points ** 2 + y_points ** 2 + z_points ** 2)

        mask = range_xyz > self.config.close_point
        x_points = x_points[mask]
        y_points = y_points[mask]
        z_points = z_points[mask]
        intensity = intensity[mask]
        range_xyz = range_xyz[mask]

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

        proj_range = np.full((height, width), -1, dtype=np.float32)
        proj_intensity = np.full((height, width), -1, dtype=np.float32)

        # Range and intensity scaling.
        range_xyz = self.range_log_base * np.log(
            (range_xyz - self.range_lower_end) * self.config.flatness_range)

        good_points_mask = intensity > self.config.min_intensity
        intensity[~good_points_mask] = 0.0
        intensity[good_points_mask] = self.intensity_log_base * np.log(
            (intensity[good_points_mask] - self.intensity_lower_end) *
            self.config.flatness_intensity)

        proj_range[proj_y, proj_x] = range_xyz
        proj_intensity[proj_y, proj_x] = np.clip(intensity, 0, 255)

        # Mask of pixels that need inpainting
        inpaint_mask = np.full((height, width), 0, dtype=np.uint8)
        inpaint_mask[proj_range <= 0] = 255

        return proj_range, proj_intensity.astype(np.uint8), inpaint_mask
