import torch
from torchvision import transforms
import timm
import cv2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from PIL import Image

import io

from generate_patches import partition_image
from usable_bark_sorter import sort_bark_from_patch_dict as sort_bark

#src = "dummy_dataset/0001/20230106_113648.jpg"
src = "white.jpg"
patches = partition_image(src, "bean_patches", 250, 250, save=True)
#accepted_patches = sort_bark(patches)

for fn, image in zip(patches.keys(), patches.values()):
    patches[fn] = Image.fromarray(image, 'RGB')

accepted_patches = sort_bark(patches)
for _, conf in accepted_patches.values():
    print(conf)
print(len(accepted_patches))
print(f"{(len(accepted_patches)/len(patches))*100:.2f}% accepted")

model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=21)
model.load_state_dict(torch.load("all_0.99-0.80_roc.bin", map_location=torch.device("cpu")))

def transform_image(image_bytes):
    ueg = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()
                            ])
    return ueg(image_bytes).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    #tensor = tensor.to(device)
    output = model.forward(tensor)

    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), classes.item()

p = []
c = []
for patch, _ in accepted_patches.values():
    conf, y_pre = get_prediction(image_bytes=patch)
    p.append(y_pre)
    c.append(conf)

votes = {}
for i, j in zip(p, c):
    votes[i] = [j]
    print(f"Predicted {i} at {j} confidence")

for i, j in zip(votes.keys(), votes.values()):
    print(f"{i}: {sum(j)/len(j)}")
