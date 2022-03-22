#!/usr/bin/env zsh

# Install required dependencies
apt -y install python3 python3-pip
apt -y install libjpeg-dev zlib1g-dev
pip3 install tqdm pillow numpy matplotlib scipy
pip3 install pytorch torchvision 
