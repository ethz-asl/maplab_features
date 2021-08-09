#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as tvf

module_path = os.path.abspath(os.path.join('extractors/r2d2'))
if module_path not in sys.path:
    sys.path.append(module_path)

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *

class Config ():
    def __init__(self):
        self.gpu = [0]
        self.model_path = os.path.join(module_path, 'models/r2d2_WASF_N16.pt')

        self.num_keypoints = 2000
        self.scale_f = 2**0.25
        self.min_scale = 1
        self.max_scale = 1
        self.min_size = 256
        self.max_size = 1024

        self.reliability_thr = 0.7
        self.repeatability_thr = 0.7

def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()

class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]

def extract_multiscale( net, img, detector, scale_f=2**0.25,
                        min_scale=0.0, max_scale=1,
                        min_size=256, max_size=1024,
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0 # current scale factor

    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])

            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores

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
        # Network related initialization
        self.config = Config()
        self.iscuda = common.torch_set_gpu(self.config.gpu)

        self.net = load_network(self.config.model_path)
        if self.iscuda:
            self.net = self.net.cuda()

        self.detector = NonMaxSuppression(
            rel_thr = self.config.reliability_thr,
            rep_thr = self.config.repeatability_thr)

        # Pipe for transferring images
        self.fifo_images = open_fifo('/tmp/maplab_features_images', 'rb')
        self.fifo_descriptors = open_fifo(
            '/tmp/maplab_features_descriptors', 'wb')

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
