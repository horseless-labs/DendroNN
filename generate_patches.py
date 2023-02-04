"""
This code is meant to take full-sized images and divide them so that they patches can be used
for training deep learning models. This is in lieu of compressing full images down to sizes of
e.g. 224x224. THIS COMPRESSION MAY NOT BE SUITABLE FOR ALL DEEP LEARNING APPLICATIONS, and is
highly likely to be specific to the problem at hand.
"""

import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=int, default=250,
        help="Reduce the images to a collection of this NxN patches")
parser.add_argument("--src", required=True, help="Source file or directory")
parser.add_argument("--dest", required=True, help="Destination directory")

args = parser.parse_args()
