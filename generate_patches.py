"""
This code is meant to take full-sized images and divide them so that they patches can be used
for training deep learning models. This is in lieu of compressing full images down to sizes of
e.g. 224x224. THIS COMPRESSION MAY NOT BE SUITABLE FOR ALL DEEP LEARNING APPLICATIONS, and is
highly likely to be specific to the problem at hand.
"""

import cv2
import shutil
import os
import argparse

parser = argparse.ArgumentParser()
# Required arguments
parser.add_argument("--src", required=True, help="Source file or directory")
parser.add_argument("--dest", required=True, help="Destination directory")

# Optional arguments
parser.add_argument("-c", action="store_true", help="Compress the output directory")
parser.add_argument("-o", action="store_true", help="Overwrite existing files")
parser.add_argument("-r", action="store_true", help="Recursively partition directories")
parser.add_argument("-s", type=int, default=250,
        help="Reduce the images to a collection of this NxN patches")
parser.add_argument("-v", action="store_true", help="Verbose")
args = parser.parse_args()

"""
Generate patches for a single image
Currently assumes the image can be divided into an integer number of patches, i.e. an
image of 4000x3000 reduced to a collection of 196 250x250 patches.
As of this writing (2023-02-04), an image that does not fit these parameters will be truncated.
An image of 1440x2560 will be reduced to a grid of 9x5 patches

TODO: write algorithm for non-integer partitions.
"""
def partition_image(image_path, dest_dir, width_cutoff, height_cutoff):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    #img = cv2.imread(source_dir + '/' + image_fn)
    image_fn = image_path.split('/')[-1]

    # Test that the file is an image.
    file_types = ["JPG", "JPEG", "jpg", "jpeg"]
    if image_fn.split(".")[-1] not in file_types:
        if args.v: print(f"{image_fn} is not an image. Skipping.")
        return

    img = cv2.imread(image_path)

    height = img.shape[0]
    width = img.shape[1]

    for i in range(int(height/height_cutoff)):
        top_border = i * height_cutoff
        bottom_border = i * height_cutoff + height_cutoff

        for j in range(int(width/width_cutoff)):
            left_border = j * width_cutoff
            right_border = j * width_cutoff + width_cutoff

            patch = img[top_border:bottom_border, left_border:right_border]
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)

            # No idea why it just goes back and forth like this. Test it
            image = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
            fn = f"{dest_dir}/{image_fn}_{i}_{j}.jpg"
            
            if os.path.exists(fn) == False or args.o:
                if args.v: print(f"Writing file {fn}")
                cv2.imwrite(fn, image)
            else:
                print("File exists and overwrite is disabled. Skipping")
                return

    # Make a ZIP archive
    if args.c: shutil.make_archive(dest_dir, "zip", dest_dir)

# This part of the code is specific to the structure of this project.
# Focused on a specimen numbering pattern with leading digits up to 9999.
def partition_directory(source_dir, dest_dir, width_cutoff, height_cutoff):
    paths = os.listdir(source_dir)

    # Supply full paths for images for easy operation
    paths = [source_dir + '/' + i for i in paths]
    paths.sort()

    # Separate sub-directories from images
    images = []
    directories = []
    for path in paths:
        if os.path.isdir(path):
            directories.append(path)
        if os.path.isfile(path):
            images.append(path)

    for image in images:
        partition_image(image, dest_dir, width_cutoff, height_cutoff)

    # Recursively navigate directories and partition those images
    if args.r:
        for directory in directories:
            if args.v: print(f"Partitioning sub-directory {directory}")
            dir_split = directory.split('/')
            dest = dest_dir + '/' + dir_split[2]
            partition_directory(directory, dest, args.s, args.s)

            # Make a ZIP archive if that option is selected
            if args.c: shutil.make_archive(dest, "zip", dest)
    else:
        print("Ignoring sub-directory {directory}")

if __name__ == "__main__":
    if os.path.isfile(args.src):
        if args.v: print("Partitioning a single image")
        partition_image(args.src, args.dest, args.s, args.s)
    if os.path.isdir(args.src):
        if args.v: print("Partitioning a directory")
        partition_directory(args.src, args.dest, args.s, args.s)
