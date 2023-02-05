"""
Based on both visual validation and experimentation with training, it made sense to only accept
bark patches that the usable bark classifier was very confident was actually bark (>=99%). That
said, a user may sometimes have different needs, and this code gives some control over the
confidence of the data that is being partitioned.

Something more relevant to my needs now is the ability to craft differnet arrangements of the data,
e.g. alternating between family and species level, only using three different members, generating a
dataset for binary classification of maple or not, and such things. This is also meant to give a
user control over such arrangements.
"""

import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', default=False, help="Above threshold? Default False.")
parser.add_argument('-c', default=0.99, help="Threshold of confidence, default of 99% (0.99)")
parser.add_argument('-l', default=0, help="Level of organization to return (0 for species, 1 for family ")
parser.add_argument('-s', default="specimen_list.csv", help="CSV file connecting specimen ID numbers to identifying information")
parser.add_argument('-v', action="store_true", help="Verbose")
args = parser.parse_args()

# Read the specimen list into a pandas DataFrame
specimen_df = pd.read_csv(args.s)
if args.v: print(specimen_df)

level = {0: 'common_name',
         1: 'family'}

level = level[int(args.l)]

# Create a dictionary with integer IDs and specimen type. Ex:
# {1: 'norway_maple'}
dict_df = specimen_df[['id', level]].copy()
dict_df.drop_duplicates(inplace=True)
dict_df.set_index('id', drop=True, inplace=True)

specimen_to_index = dict_df.to_dict()[level]
if args.v:
    for i in specimen_to_index:
        print(i, specimen_to_index[i])

members = set(specimen_to_index.values())
print(members)

def make_selection(df, mode, members):
    members = sorted(list(members))
    text = ""
    for i in range(len(members)):
        text += f"{i}: {members[i]}\n"

    # Of the type "maple or not"
    if mode == "one_in_all":
        print(text)
        selection = input("Enter the number to single out: ")
        print(f"Singling out {members[int(selection)]}")
    # Of the type "maple, beech, elm, and ..."
    if mode == "selection":
        print(text)
        print("Enter the selection you want to keep, separated by commas.")
        selection = input("e.g 14, 9, 17\n ")

        dfs = []

        selection = selection.split(', ')
        selection = [int(i) for i in selection]
        for i in selection:
            dfs.append(df[df[level] == members[i]])
            print(f"Keeping {members[i]}")

    if mode == "all":
        print("All members selected")

make_selection("selection", members)

# Returns all members of the dataset on either side of a confidence threshold.
# df here is a pandas DataFrame
def confidence_threshold(df, thresh=args.c, below=args.b):
    if below == True:
        return df[df['confidence'] < thresh]
    else:
        return df[df['confidence'] >= thresh]

def image_reshape(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))
    image = np.asarray(image)
    return image
