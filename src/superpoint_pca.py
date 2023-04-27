#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import csv
from sklearn.decomposition import PCA

N_COMPONENTS = 64

module_path = os.path.join(os.path.dirname(__file__), 'trackers/superglue')
if module_path not in sys.path:
    sys.path.append(module_path)

from models.superpoint import SuperPoint

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

# Get image list path
if len(sys.argv) != 2:
    exit("Usage: python3 superpoint_pca.py <path-to-image-list>")

# Network related initialization
torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

superpoint = SuperPoint({
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024
}).eval().to(device)

# Image list
images = []
with open(sys.argv[1], "r") as fp:
    for line in fp:
        images.append(line.strip())

descriptors = []
for i, image_path in enumerate(images):
    cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv_image = frame2tensor(cv_image.astype(np.float32), device)

    # Get superpoint output
    pred = superpoint({'image': cv_image})
    desc = pred['descriptors'][0].cpu().numpy()
    desc = desc.transpose((1, 0))
    descriptors.append(desc)

    if i % 100 == 0:
        print(i)
descriptors = np.concatenate(descriptors, axis=0)

pca = PCA(n_components=N_COMPONENTS)
pca.fit(descriptors)
print("Explained", np.sum(pca.explained_variance_ratio_))

with open("pca.txt", "w") as fp:
    mean = pca.mean_
    components = pca.components_.T
    writer = csv.writer(fp, delimiter = ',')
    writer.writerow(components.shape)
    writer.writerow(mean)
    writer.writerows(components)
