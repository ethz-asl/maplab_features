#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch

module_path = os.path.abspath(os.path.join('trackers/superglue'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.superpoint import SuperPoint
from models.utils import frame2tensor

torch.set_grad_enabled(False)

def open_fifo(file_name, mode):
    try:
        os.mkfifo(file_name)
    except FileExistsError:
        pass
    return open(file_name, mode)

def read_bytes(file, num_bytes):
    bytes = b''
    num_read = 0
    while num_read < num_bytes:
        bytes += file.read(num_bytes - num_read)
        num_read = len(bytes)
    return bytes

def read_np(file, dtype):
    num_bytes = read_bytes(file, 4)
    num_bytes = np.frombuffer(num_bytes, dtype=np.uint32)[0]
    bytes = read_bytes(file, num_bytes)
    return np.frombuffer(bytes, dtype=dtype)

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

    def to_torch(self, arr):
        return torch.from_numpy(arr).float()[None].to(self.device)

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

        # Transmit number of data bytes as well as number of detected keypoints
        # and the descriptor size to reshape the array
        scores = np.expand_dims(scores, axis=1)
        scales = np.expand_dims(scales, axis=1)

        descriptor_data = np.concatenate(
            [xy, scores, scales, descriptors],
            axis=1).astype(np.float32).tobytes()
        num_bytes = len(descriptor_data)
        num_keypoints = descriptors.shape[0]
        descriptor_size = descriptors.shape[1]
        descriptor_header = np.array(
            [num_bytes, num_keypoints, descriptor_size],
            dtype=np.uint32).tobytes()

        self.fifo_descriptors.write(descriptor_header)
        self.fifo_descriptors.write(descriptor_data)
        self.fifo_descriptors.flush()

def main():
    image_receiver = ImageReceiver()

    while True:
        image_receiver.callback()

if __name__ == '__main__':
    main()
