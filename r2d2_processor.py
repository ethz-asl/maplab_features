#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np

from r2d2_interface import *

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

class ImageReceiver:
    def __init__(self):
        # Pipe for transferring images
        self.fifo_images = open_fifo('/tmp/maplab_features_images', 'rb')
        self.fifo_descriptors = open_fifo(
            '/tmp/maplab_features_descriptors', 'wb')

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
        num_bytes = read_bytes(self.fifo_images, 4)
        num_bytes = np.frombuffer(num_bytes, dtype=np.uint32)[0]
        cv_binary = read_bytes(self.fifo_images, num_bytes)
        cv_image = cv2.imdecode(np.frombuffer(
            cv_binary, dtype=np.uint8), flags=1)

        # Prepare image for processing
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        W, H, _ = image.shape
        image = norm_RGB(image)[None]
        if self.iscuda:
            image = image.cuda()

        # Extract keypoints
        xys, descriptors, scores = extract_multiscale(
            self.net, image, self.detector,
            scale_f   = self.config.scale_f,
            min_scale = self.config.min_scale,
            max_scale = self.config.max_scale,
            min_size  = self.config.min_size,
            max_size  = self.config.max_size)

        xys = xys.cpu().numpy()
        descriptors = descriptors.cpu().numpy()
        scores = scores.cpu().numpy()

        idxs = scores.argsort()[-self.config.num_keypoints or None:]
        xy = xys[idxs, :2]
        descriptors = descriptors[idxs]
        scores = np.expand_dims(scores[idxs], axis=1)
        scales = np.expand_dims(xys[idxs, 2], axis=1)

        # Transmit number of data bytes as well as number of detected keypoints
        # and the descriptor size to reshape the array
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

        #for kp in xy:
        #    cv2.circle(cv_image, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)

def main():
    image_receiver = ImageReceiver()

    while True:
        image_receiver.callback()

if __name__ == '__main__':
    main()
