import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import timm

import numpy as np
import pandas as pd

from PIL import Image

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import io

from collections import defaultdict

import os, copy, time, gc, re

import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-b', default=8, help="Batch size")
parser.add_argument("-c", help="Base name of train and test CSV files")
parser.add_argument("-d", action="store_true", help="Developer mode")
parser.add_argument("-e", default=20, help="Number of epochs")
parser.add_argument("-f", default=1.0, help="Fraction of available data to train and test with.")
parser.add_argument("-l", default="common_name", help="Level of specimen's organization, e.g. family, common_name")
parser.add_argument("-m", default="deit_base_patch16_224", help="Name of timm model to load")
parser.add_argument("-s", default=224, help="Size to scale image down to")
parser.add_argument("-v", action="store_true", help="Verbose")
parser.add_argument("-w", default="", help="Path to weights file")
parser.add_argument("--test_only", action="store_true", help="Does not train the model.")
parser.add_argument("--log", action="store_true", help="Enables logging mode")
args = parser.parse_args()

EPOCHS = int(args.e)
BATCH_SIZE = int(args.b)
SIZE = int(args.s)
FRAC = float(args.f)
level = args.l

dataset_dir = "dataset/"
#csv_base = "bur_oak+chestnut_oak+northern_red_oak+pin_oak+swamp_white_oak+white_oak-0.99"
#csv_base = "bur_oak+pin_oak-0.99"
csv_base = args.c
train_file = dataset_dir + f"{csv_base}_train.csv"
test_file = dataset_dir + f"{csv_base}_test.csv"

# 6-class oak
model_weights = args.w
model_name = args.m

# Binary
#model_weights = "deit_tiny-bur_oak+pin_oak-12_epochs-99.2_auroc.bin"

train_df = pd.read_csv(train_file)
train_df["path"] = dataset_dir + train_df["path"]
train_df["factor"] = pd.factorize(train_df[level])[0]

test_df = pd.read_csv(test_file)

label_df = copy.deepcopy(train_df)
label_df["label"] = LabelEncoder().fit_transform(train_df["factor"])

dict_df = train_df[[level, "factor"]].copy()
dict_df.drop_duplicates(inplace=True)
dict_df.set_index(level, drop=True, inplace=True)
factor_to_index = dict_df.to_dict()["factor"]
index_to_factor = {y: x for x, y in factor_to_index.items()}

skf = StratifiedKFold(n_splits=8)

for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.factor)):
        train_df.loc[val_, "kfold"] = fold

# Fixing errors in the dataset
# Remove when the dataset is finalized
train_df["path"] = train_df["path"].str.replace("dataset/dataset0/", "dataset/")
test_df["path"] = test_df["path"].str.replace("dataset0/", "dataset/")

if FRAC < 1.0:
    train_df = train_df.sample(frac=FRAC)
    test_df = test_df.sample(frac=FRAC)

N_CLASSES = len(train_df["factor"].unique())

class BarkDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.fns = df["path"].values
        self.labels = df["factor"].values

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.25, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row.path)
        label = self.labels[idx]

        image = self.train_transforms(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(df, fold):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = BarkDataset(train_df)
    valid_dataset = BarkDataset(valid_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4)
    
    return train_loader, valid_loader

def transform_image(image_bytes):
    ueg = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                            transforms.CenterCrop(SIZE),
                            transforms.ToTensor(),
                            ])
    image = Image.open(io.BytesIO(image_bytes))
    return ueg(image).unsqueeze(0)

# TODO: modify this to return e.g. top-3 or top-5
def single_prediction(image_bytes, model):
    model.eval()

    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.to(device)
    output = model.forward(tensor)

    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), classes.item()

def multiclass_roc(actual_class, pred_class, average="macro"):
    unique_class = set(actual_class)

    roc_auc_dict = {}
    for per_class in unique_class:
        # Create list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # Marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # Calculate ROC
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

def testing(test_df, model):
    image_paths = test_df["path"].values.tolist()
    correct_labels = test_df[level].values.tolist()

    # Labels for confusion matrix
    # TODO: fix this
    #factors = {y: x for x, y in factor_to_index.items()}
    #print(factors.values())

    # Predictions and confidence values
    preds = []
    confs = []
    count = 0
    for image_path in tqdm(image_paths):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            conf, y_pre = single_prediction(image_bytes=image_bytes, model=model)
            confs.append(conf)
            preds.append(y_pre)
            count += 1

    targets = []
    for i in correct_labels:
        targets.append(factor_to_index[i])

    scores = multiclass_roc(targets, preds)
    rocs = list(scores.values())

    classes = list(scores.keys())
    print(scores)
    print(classes)
    classes = {index_to_factor[i]: scores[i] for i in range(len(classes))}

    mean_roc = np.sum(rocs)/len(rocs)

    matrix = confusion_matrix(targets, preds)
    return mean_roc, matrix, classes

def train_epoch(model, optimizer, criterion, scheduler, dataloader, device, epoch):
    model.train()
            
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        labels = data["label"].to(device, dtype=torch.long)

        batch_size = images.size(0)
         
        #outputs, emb = model(images, labels)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
         
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
         
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
                        
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Train_loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
        
    gc.collect()
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        labels = data["label"].to(device, dtype=torch.long)

        batch_size = images.size(0)

        #outputs, emb = model(images, labels)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Valid_loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    gc.collect()
    return epoch_loss

def time_elapsed(start, end):
    elapsed = end - start
    elapsed_str = "{:.0f}h {:.0f}m {:.0f}s".format(elapsed // 3600,
                                (elapsed % 3600) // 60, (elapsed % 3600) % 60)
    return elapsed_str


def training(model, optimizer, criterion, scheduler, device, num_epochs):
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch_loss = 1000000
    history = defaultdict(list)
    model_fns = []

    train_loader, valid_loader = get_dataloaders(train_df, fold=0)

    epochs_since_improvement = 0
    for epoch in range(1, num_epochs+1):
        gc.collect()
        train_epoch_loss = train_epoch(model, optimizer, criterion, scheduler,
                                        dataloader=train_loader, device=device, epoch=epoch)
        val_epoch_loss = validate_epoch(model, valid_loader, criterion, device=device, epoch=epoch)

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)
        
        if val_epoch_loss <= best_epoch_loss:
            print(f"New best validation: {val_epoch_loss}")
            best_epoch_loss = val_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())

            PATH = "loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            model_fns.append(PATH)

            epochs_since_improvement = 0
            print("Model saved")
        else:
            epochs_since_improvement += 1
        print()
        end = time.time()
        print(f"Training complete in {time_elapsed(start, end)}")
        print("Best loss: {:.4f}".format(best_epoch_loss))
        model.load_state_dict(best_model_weights)
        if epochs_since_improvement == 5:
            print("Stopping early due to poor improvement. Goodbye.")
            break
    return model, history, model_fns

if __name__ == '__main__':
    # Reduce size of dataset for developer mode
    if args.d:
        EPOCHS = 2
        train_df = train_df.sample(frac=0.001)
        test_df = test_df.sample(frac=0.1)

    model = timm.create_model(model_name, pretrained=True, num_classes=N_CLASSES)
    if model_weights != "":
        model.load_state_dict(torch.load(model_weights))
    model.to(device)

    
    start_time = time.ctime()
    start = time.time()
    if not args.test_only:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1,
                                                        eta_min=0.0000, last_epoch=-1)
        model, history, model_fns = training(model, optimizer, criterion, scheduler, device=device, num_epochs=EPOCHS)
    stop = time.time()
    elapsed_time = time_elapsed(start, stop)

    roc, conf_matrix, classes = testing(test_df, model)

    # Get the actual number of trained epochs in the last model with improvement
    log_text = "\n"
    log_text += "*" * 32
    if not args.test_only:
        log_text += f"\nTraining session started on {start_time}\n"
        # Get the actual number of trained epochs in the last model with improvement
        epoch = model_fns[-1]
        epoch = epoch.split('_')[1]
        epoch = int(re.findall("\d+", epoch)[0])

        log_text += f"Model: {args.m}"
        if model_weights != "":
            log_text += f" using weights {model_weights}\n"
        else:
            log_text += ", starting with just timm weights\n"

        log_text += f"CSV base: {args.c}, \n{FRAC*100}% at the {args.l} level of organization\n"
        log_text += f"{EPOCHS} epochs attempted, {epoch} run\n"
        log_text += f"This training session generated new weights at: {model_fns[-1]}\n"

    log_text += f"\nTesting session started on {time.ctime()}\n"
    if args.test_only:
        log_text += f"Model: {args.m} using weights {model_weights}\n"
    log_text += f"The confusion matrix is as follows:\n\n{conf_matrix}\n\n"
    log_text += f"Mean ROC: {roc}, or by class: \n\n"

    for class_roc in classes:
        cls = f"{class_roc}"
        roc = f"{classes[class_roc]:.3f}"
        dots = "." * (45 - len(cls) - len(roc))
        
        log_text += cls + dots + roc + "\n"
        #log_text += f"{class_roc}: {classes[class_roc]:.3f}\n"
    log_text += "\n"

    if args.v: print(log_text)
    if args.log:
        with open('log.txt', 'a+') as f:
            f.write(log_text)
        print("Logged")
