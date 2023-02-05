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
parser.add_argument('-a', default="torso_accept.csv", help="CSV file of accepted bark patches")
parser.add_argument('-b', default=False, help="Above threshold? Default False.")
parser.add_argument('-c', default=0.99, help="Threshold of confidence, default of 99% (0.99)")
parser.add_argument('-l', default=0, help="Level of organization to return (0 for species, 1 for family ")
parser.add_argument('-s', default="specimen_list.csv", help="CSV file connecting specimen ID numbers to identifying information")
parser.add_argument('-v', action="store_true", help="Verbose")
args = parser.parse_args()

# Read the specimen list into a pandas DataFrame
specimen_df = pd.read_csv(args.s)
if args.v: print(specimen_df)

# Select the level of organization
level = {0: "common_name",
         1: "family"}

level = level[int(args.l)]

# Open the file of accepted bark patches
accept_fn = args.a
accept_df = pd.read_csv(accept_fn, index_col=0)
if "path" and "confidence" not in accept_df.columns:
    print(f"Column names in {accept_fn} do not match requirements to generate training")
    print("and test files. Need 'path' and 'confidence' columns. Did you load the wrong one?")
    exit()

# Create a dictionary with integer IDs and specimen type. Ex:
# {1: 'norway_maple'}
dict_df = specimen_df[['id', level]].copy()
dict_df.drop_duplicates(inplace=True)
dict_df.set_index('id', drop=True, inplace=True)

specimen_to_index = dict_df.to_dict()[level]
if args.v:
    for i in specimen_to_index:
        print(i, specimen_to_index[i])

# Extract the specimen name, remove leading zeros, and add the level
df = accept_df.copy()
df["specimen"] = [i.split('/')[2] for i in df['path']]
df["specimen"] = df["specimen"].str.lstrip('0')
df[level] = [specimen_to_index[int(df.iloc[i].specimen)] for i in range(len(df))]
print(df.head())

members = set(specimen_to_index.values())
print(members)

# Returns all members of the dataset on either side of a confidence threshold.
# df here is a pandas DataFrame
def confidence_threshold(df, thresh=args.c, below=args.b):
    if below == True:
        return df[df['confidence'] < thresh]
    else:
        return df[df['confidence'] >= thresh]

def make_selection(df, mode, members, frac=args.c):
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
            print(f"Keeping {members[i]}")
            ueg_member = df[df[level] == members[i]]
            ueg_member = confidence_threshold(ueg_member)
            dfs.append(ueg_member)

        print(dfs)

    if mode == "all":
        print("All members selected")

make_selection(df, "selection", members)
