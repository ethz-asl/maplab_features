#!/usr/bin/env python2

import numpy as np
import sensor_msgs.point_cloud2 as pc2

def convert_msg_to_array(pcl_msg):
    points_list = []
    for data in pc2.read_points(pcl_msg, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])
    return np.array(points_list)

# Based on https://github.com/PRBonn/OverlapNet/blob/master/src/utils/utils.py
def project_cloud_to_2d(cloud, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=200):
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(cloud[:, :3], 2, axis=1)
    mask = (depth > 0) & (depth < max_range)
    cloud = cloud[mask]
    depth = depth[mask]

    # get scan components
    scan_x = cloud[:, 0]
    scan_y = cloud[:, 1]
    scan_z = cloud[:, 2]
    intensity = cloud[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    proj_range = np.full((proj_H, proj_W), -1,
                   dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                 dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_intensity[proj_y, proj_x] = intensity
    return proj_range, proj_intensity