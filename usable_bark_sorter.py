import torch
from torchvision import transforms, models, datasets

import pandas as pd

import timm

from PIL import Image

from tqdm import tqdm
import os, io, time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = "dummy_dataset/"
N_CLASSES = 2
specimens = os.listdir(PATH)

yesno = {0: "accept",
        1: "reject"}

model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=N_CLASSES)
model.load_state_dict(torch.load("usable_bark_sorter.bin", map_location=torch.device(device)))
model.to(device)

def transform_image(image_bytes, patch=False):
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    if patch == False:
        print("patch == False")
        image_bytes = Image.open(io.BytesIO(image_bytes))

    return trans(image_bytes).unsqueeze(0)

def single_prediction(image_bytes, model, patch=False):
    tensor = transform_image(image_bytes=image_bytes, patch=patch)
    tensor = tensor.to(device)
    output = model(tensor)

    probs = torch.nn.functional.softmax(output, dim=1)

    conf, classes = torch.max(probs, 1)
    return conf.item(), yesno[classes.item()]

# Extract the usable bark from an entire dataset.
def sort_bark_from_path():
    accept_paths, reject_paths = [], []
    accept_conf, reject_conf = [], []
    accept_count, reject_count = 0, 0

    for specimen in tqdm(sorted(specimens)):
        print(f"Specimen {specimen}")
        start = time.time()
        src = PATH + specimen + '/'
        fns = os.listdir(src)

        for fn in tqdm(fns):
            with open(src + fn, 'rb') as f:
                image_bytes = f.read()
                conf, y_pred = single_prediction(image_bytes=image_bytes, model=model)

                if y_pred == "reject":
                    reject_count += 1
                    reject_paths.append(src+fn)
                    reject_conf.append(conf)
                else:
                    accept_count += 1
                    accept_paths.append(src+fn)
                    accept_conf.append(conf)

        end = time.time()
        print(f"{specimen} took {(end-start)/60:.2f} minutes to sort\n")

    total = reject_count + accept_count
    acceptance = accept_count/total * 100.0
    print(f"{acceptance}% accepted")

    accept_df = pd.DataFrame(list(zip(accept_paths, accept_conf)), columns=["path", "confidence"])
    reject_df = pd.DataFrame(list(zip(reject_paths, reject_conf)), columns=["path", "confidence"])
    return accept_df, reject_df

def sort_bark_from_patch_dict(patch_dict, thresh=0.90):
    accepted_patches = {}
    for fn, image in tqdm(zip(patch_dict.keys(), patch_dict.values())):
        conf, y_pred = single_prediction(image_bytes=image, model=model, patch=True)

        print(conf)
        if y_pred == "accept" and conf >= thresh:
            accepted_patches[fn] = image, conf

    return accepted_patches

if __name__ == '__main__':
    #accept_df, reject_df = sort_bark_from_path()
    #sort_bark_from_patch_dict(patch_dict)
    pass
