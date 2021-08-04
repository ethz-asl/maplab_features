#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np

from r2d2_interface import *

class ImageReceiver:
    def __init__(self):
        # Pipe for transferring images
        self.fifo_file = '/tmp/image_pipe'
        try:
            os.mkfifo(self.fifo_file)
        except FileExistsError:
            pass

        # Network related initialization
        self.config = Config()
        self.iscuda = common.torch_set_gpu(self.config.gpu)

        self.net = load_network(self.config.model_path)
        if self.iscuda:
            self.net = self.net.cuda()

        self.detector = NonMaxSuppression(
            rel_thr = self.config.reliability_thr,
            rep_thr = self.config.repeatability_thr)

    def callback(self):
        # Receive image on pipe and decode
        with open(self.fifo_file, 'rb') as fifo:
            cv_binary = fifo.read()

        cv_image = cv2.imdecode(np.frombuffer(
                cv_binary, dtype=np.uint8), flags=1)

        # Prepare image for processing
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        W, H, _ = image.shape
        image = norm_RGB(image)[None]
        if self.iscuda:
            image = image.cuda()

        # Extract keypoints
        xys, desc, scores = extract_multiscale(self.net, image, self.detector,
            scale_f   = self.config.scale_f,
            min_scale = self.config.min_scale,
            max_scale = self.config.max_scale,
            min_size  = self.config.min_size,
            max_size  = self.config.max_size)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()

        idxs = scores.argsort()[-self.config.num_keypoints or None:]
        xys = xys[idxs]
        desc = desc[idxs]
        scores = scores[idxs]

        #for kp in xys:
        #    cv2.circle(cv_image, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)

def main():
    image_receiver = ImageReceiver()

    while True:
        image_receiver.callback()

if __name__ == '__main__':
    main()
