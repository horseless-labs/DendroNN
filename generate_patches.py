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

# Generate patches for a single image
# Currently assumes the image can be divided into an integer number of patches, i.e. an
# image of 4000x3000 reduced to a collection of 196 250x250 patches.
# TODO: write algorithm for non-integer partitions.
def partition_image(image_fn, source_dir, dest_dir, width_cutoff, heigh_cutoff):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    img = cv2.imread(source_dir + image_fn)

    height = img.shape[0]
    width = img.shape[1]

    for i in range(int(height/height_cutoff)):
        top_border = i * height_cutoff
        bottom_border = i * height_cutoff + height_cutoff

        for j in range(int(width/width_cutoff)):
            left_border = j * width_cutoff
            right_border = j * width_cutoff + width_cutoff

            patch = img[top_border:bottom_border, left_border:right_border]
            patch = cv2.rotate(section, cv2.ROTATE_90_CLOCKWISE)

            # No idea why it just goes back and forth like this. Test it
            image = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
            fn = f"{dest_dir}/{image_fn}_{i}_{j}.jpg"
            cv2.imwrite(fn, image)
