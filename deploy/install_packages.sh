#!/usr/bin/env zsh

# Install required dependencies
apt -y install python3 python3-pip
apt -y install libjpeg-dev zlib1g-dev
apt -y install ffmpeg libsm6 libxext6

pip3 install --upgrade pip
pip3 install opencv-python
pip3 install tqdm pillow numpy matplotlib scipy
pip3 install torch torchvision
