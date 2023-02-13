import numpy as np
from PIL import Image
import cv2

import os

path = "dummy_dataset/0001_w_blanks/"

fn = os.listdir(path)[0]

def get_base_fn(fn):
    base_fn = fn.split(".")[0]
    pieces = base_fn.split("_")
    return pieces[0] + "_" + pieces[1]

# Presumes a set of 16x12 patches
# Checks a given base file name to see where patches are available.
def patch_grid(base_fn):
    missing = []
    available = []
    for i in range(16):
        for j in range(12):
            prov_fn = f"{path+base_fn}_{i}_{j}.jpg"
            if os.path.isfile(f"{path+base_fn}_{i}_{j}.jpg"):
                available.append([i, j])
                #available.append(prov_fn)
            else:
                #available.append(['na', 'na'])
                missing.append([i, j])
    return missing, available

fn = get_base_fn(fn)
missing, available = patch_grid(fn)
available = np.asarray(available)

"""
print("The following patches were missing and need to be rendered as black.")
for i in missing:
    print(f"{path+fn}_{i[0]}_{i[1]}.jpg")
"""

blank_image = np.zeros((250, 250, 3))
rows = []
for i in range(16):
    row = []
    for j in range(12):
        prov_fn = f"{path+fn}_{i}_{j}.jpg"
        print(prov_fn)
        if os.path.isfile(prov_fn):
            image = cv2.imread(prov_fn)
            row.append(image)
        else:
            row.append(blank_image)
    row = np.asarray(row)
    row = cv2.hconcat(row)
    rows.append(row)

rows = np.asarray(rows)
rows = cv2.vconcat(rows)
print(rows.shape)
cv2.imwrite("reassembled.jpg", rows)
