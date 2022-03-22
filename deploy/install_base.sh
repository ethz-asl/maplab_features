#!/usr/bin/env zsh

# Base install
apt update
apt -y upgrade
apt -y install software-properties-common \
  python2.7 \
  python3 \
  python-pip \
  python-termcolor \
  git \
  curl \
  iproute2
