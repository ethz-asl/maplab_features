#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch

from utils_py3 import open_fifo, read_np, send_np

module_path = os.path.abspath(os.path.join('trackers/superglue'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.superpoint import SuperPoint
from models.utils import frame2tensor

torch.set_grad_enabled(False)

class ImageReceiver:
    def __init__(self):
        # Network related initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.superpoint = SuperPoint({
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }).eval().to(self.device)

        # Pipe for transferring detection
        self.fifo_images = open_fifo(
            '/tmp/maplab_features_images', 'rb')
        self.fifo_descriptors = open_fifo(
            '/tmp/maplab_features_descriptors', 'wb')

    def callback(self):
        # Receive image on pipe and decode
        cv_image = cv2.imdecode(read_np(self.fifo_images, np.uint8),
            flags=cv2.IMREAD_GRAYSCALE)

        # Preprocess for pytorch
        cv_image = frame2tensor(cv_image.astype(np.float32), self.device)

        # Get superpoint output
        pred = self.superpoint({'image': cv_image})
        xy = pred['keypoints'][0].cpu().numpy()
        scores = pred['scores'][0].cpu().numpy()
        scales = np.zeros(xy.shape[0])
        descriptors = pred['descriptors'][0].cpu().numpy()
        descriptors = descriptors.transpose((1, 0))

        # Transmit extracted descriptors back
        send_np(self.fifo_descriptors, xy.astype(np.float32))
        send_np(self.fifo_descriptors, scores.astype(np.float32))
        send_np(self.fifo_descriptors, scales.astype(np.float32))
        send_np(self.fifo_descriptors, descriptors.astype(np.float32))

def main():
    image_receiver = ImageReceiver()

    while True:
        image_receiver.callback()

if __name__ == '__main__':
    main()
