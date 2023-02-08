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

import os, copy, time, gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 20
BATCH_SIZE = 8
SIZE = 224
FRAC = 1.0
level = "common_name"

dataset_dir = "dataset/"
#csv_base = "bur_oak+chestnut_oak+northern_red_oak+pin_oak+swamp_white_oak+white_oak-0.99"
#csv_base = "bur_oak+pin_oak-0.99"
csv_base = "balanced_oaks"
train_file = dataset_dir + f"{csv_base}_train.csv"
test_file = dataset_dir + f"{csv_base}_test.csv"

# 6-class oak
model_weights = ""
model_name = "deit_base_patch16_224"

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
#print(train_df)

if FRAC < 1.0:
    train_df = train.sample(frac=FRAC)

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
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
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
            """
            if count%100==0:
                print(f"{(count/len(image_paths))*100:.2f}% complete.")
            if count+1 == len(image_paths):
                print("Finished test.")
            """

            image_bytes = f.read()
            conf, y_pre = single_prediction(image_bytes=image_bytes, model=model)
            confs.append(conf)
            preds.append(y_pre)
            count += 1

    targets = []
    for i in correct_labels:
        targets.append(factor_to_index[i])

    scores = multiclass_roc(targets, preds)
    scores = list(scores.values())
    mean_score = np.sum(scores)/len(scores)

    matrix = confusion_matrix(targets, preds)
    return mean_score, matrix

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

def training(model, optimizer, criterion, scheduler, device, num_epochs):
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch_loss = 1000000
    history = defaultdict(list)

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
            epochs_since_improvement = 0
            print("Model saved")
        else:
            epochs_since_improvement += 1
        print()
        end = time.time()
        time_elapsed = end - start
        print("Training complete in {:.0f}h {:.0f}m {:.0f}s".format(time_elapsed // 3600,
                                (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best loss: {:.4f}".format(best_epoch_loss))
        model.load_state_dict(best_model_weights)
        if epochs_since_improvement == 5:
            print("Stopping early due to poor improvement. Goodbye.")
            break
    return model, history

if __name__ == '__main__':
    #train_df = train_df.sample(frac=0.01)
    #test_df = test_df.sample(frac=0.01)

    model = timm.create_model(model_name, pretrained=True, num_classes=N_CLASSES)
    if model_weights != "":
        model.load_state_dict(torch.load(model_weights))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1,
                                                        eta_min=0.00001, last_epoch=-1)
    model, history = training(model, optimizer, criterion, scheduler, device=device, num_epochs=EPOCHS)
    roc, conf_matrix = testing(test_df, model)
    print(f"The mean ROC score is: {roc}")
    print("The confusion matrix is as follows:")
    print(conf_matrix)
