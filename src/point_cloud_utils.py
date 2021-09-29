#!/usr/bin/env python2

import numpy as np
import sensor_msgs.point_cloud2 as pc2

class Utils:
    def __init__(self):

    def convert_msg_to_array(pcl_msg):
        points_list = []
        for data in pc2.read_points(pcl_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    # Based on https://github.com/PRBonn/OverlapNet/blob/master/src/utils/utils.py
    def project_cloud_to_2d(cloud, fov_up=3.0, fov_down=-25.0, height=64, width=900, max_range=200):
        fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

        # get scan components
        x_points = cloud[:, 0]
        y_points = cloud[:, 1]
        z_points = cloud[:, 2]
        range = np.sqrt(x_points ** 2 + y_points ** 2, z_points ** 2) + 1e-6 # distance to origin
        intensity = cloud[:, 3]

        # get angles of all points
        pitch = np.arcsin(np.clip(z_points/range, -1, 1))
        yaw = np.arctan2(y_points, -x_points)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= width  # in [0.0, W]
        proj_y *= height  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(width - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(height - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        proj_range = np.full((height, width), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
        proj_intensity = np.full((height, width), -1,
                     dtype=np.float32)  # [H,W] index (-1 is no data)

        proj_range[proj_y, proj_x] = range
        proj_intensity[proj_y, proj_x] = intensity
        return proj_range, proj_intensity
