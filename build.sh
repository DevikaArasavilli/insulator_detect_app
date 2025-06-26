#!/bin/bash

# Install dependencies except TensorFlow
pip install -r requirements.txt || exit 1

# Manually install a version of TensorFlow compatible with Python 3.10
pip install tensorflow==2.10.0 || exit 1
