#!/usr/bin/env zsh

# Base install
apt update
apt -y upgrade
apt -y install software-properties-common \
  python2.7 \
  python-pip \
  python-termcolor \
  git \
  curl \
  iproute2

# Base python install
pip2 isntall setuptools wheel
pip2 install numpy scipy
pip2 install opencv-python==4.2.0.32
