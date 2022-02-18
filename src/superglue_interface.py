#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.cm as cm

from utils_py2 import open_fifo, read_np, send_np

module_path = os.path.abspath(os.path.join('trackers/superglue'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.superglue import SuperGlue
from models.utils import frame2tensor, make_matching_plot

torch.set_grad_enabled(False)

class ImageReceiver:
    def __init__(self):
        # Network related initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.superglue = SuperGlue({
            'weights': 'indoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }).eval().to(self.device)

        # Pipe for transferring tracking
        self.fifo_images = open_fifo(
            '/tmp/maplab_superglue_images', 'rb')
        self.fifo_matches = open_fifo(
            '/tmp/maplab_superglue_matches', 'wb')

        self.descriptor_size = 256
        self.debug = False

    def to_torch(self, arr):
        return torch.from_numpy(arr).float()[None].to(self.device)

    def callback(self):
        # Receive image on pipe and decode
        cv_image0 = cv2.imdecode(read_np(self.fifo_images, np.uint8),
            flags=cv2.IMREAD_GRAYSCALE)
        cv_image1 = cv2.imdecode(read_np(self.fifo_images, np.uint8),
            flags=cv2.IMREAD_GRAYSCALE)

        xy0 = read_np(self.fifo_images, np.float32).reshape((-1, 2))
        scores0 = read_np(self.fifo_images, np.float32).flatten()
        descriptors0 = read_np(self.fifo_images, np.float32).reshape(
            (-1, self.descriptor_size)).transpose((1, 0))

        xy1 = read_np(self.fifo_images, np.float32).reshape((-1, 2))
        scores1 = read_np(self.fifo_images, np.float32).flatten()
        descriptors1 = read_np(self.fifo_images, np.float32).reshape(
            (-1, self.descriptor_size)).transpose((1, 0))

        # Preprocess for pytorch
        image0 = frame2tensor(cv_image0.astype(np.float32), self.device)
        image1 = frame2tensor(cv_image1.astype(np.float32), self.device)

        # Get superglue output
        data = {'image0': image0,
                'image1': image1,
                'keypoints0': self.to_torch(xy0),
                'keypoints1': self.to_torch(xy1),
                'scores0': self.to_torch(scores0),
                'scores1': self.to_torch(scores1),
                'descriptors0': self.to_torch(descriptors0),
                'descriptors1': self.to_torch(descriptors1)}

        pred = self.superglue(data)
        matches = pred['matches0'][0].cpu().numpy().astype(np.int32)
        conf = pred['matching_scores0'][0].cpu().numpy()

        send_np(self.fifo_matches, matches)

        if self.debug:
            # Keep the matching keypoints.
            valid = matches > -1
            mxy0 = xy0[valid]
            mxy1 = xy1[matches[valid]]
            mconf = conf[valid]

            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(xy0), len(xy1)),
                'Matches: {}'.format(len(mxy0)),
            ]

            make_matching_plot(
                cv_image0, cv_image1, xy0, xy1, mxy0, mxy1, color,
                text, "./match.png", True, True, True, 'Matches', '')

def main():
    image_receiver = ImageReceiver()

    while True:
        image_receiver.callback()

if __name__ == '__main__':
    main()
