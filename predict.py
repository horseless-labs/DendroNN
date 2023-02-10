import torch
import timm
import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import io

from generate_patches import partition_image
from usable_bark_sorter import sort_bark_from_patch_dict as sort_bark

src = "dummy_dataset/0001/20230106_113648.jpg"
patches = partition_image(src, "bean_patches", 250, 250, save=True)
#accepted_patches = sort_bark(patches)

for fn, image in zip(patches.keys(), patches.values()):
    patches[fn] = Image.fromarray(image, 'RGB')

accepted_patches = sort_bark(patches)
for _, conf in accepted_patches.values():
    print(conf)
print(len(accepted_patches))
print(f"{(len(accepted_patches)/len(patches))*100:.2f}% accepted")
