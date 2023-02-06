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
parser.add_argument('-m', default=0, help="Selection mode. 0 for one-in-all (e.g. binary), 1 for arbitrary selection, 2 for all (just returns on confidence threshold")
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

# Select the selection mode
mode = {0: "one_in_all",
        1: "arbitrary",
        2: "all"}
mode = mode[int(args.m)]

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
if args.v: print(df.head())

members = set(specimen_to_index.values())
if args.v: print(members)

# Returns all members of the dataset on either side of a confidence threshold.
# df here is a pandas DataFrame
def confidence_threshold(df, thresh=args.c, below=args.b):
    if below == True:
        return df[df['confidence'] < thresh]
    else:
        return df[df['confidence'] >= thresh]

# Helpfer function to sort the list of members and generate a selection menu.
def members_text(members):
    members = sorted(list(members))
    text = ""
    for i in range(len(members)):
        text += f"{i}: {members[i]}\n"
    return members, text

def one_in_all_mode(df, members, conf=args.c):
    members, text = members_text(members)
    print(text)
    selection = input("Enter the number of the member to single out: ")
    print(f"Singling out {members[int(selection)]}")

    focus = df[df[level] == members[int(selection)]]
    ignore = df[df[level] != members[int(selection)]]

    focus = confidence_threshold(focus)
    ignore = confidence_threshold(ignore)
    return focus, ignore

def arbitrary_mode(df, members, conf=args.c):
    members, text = members_text(members)
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

    return dfs

def all_mode(df, conf=args.c):
    return confidence_threshold(df)

if __name__ == '__main__':
    if mode == "one_in_all":
        focus, ignore = one_in_all_mode(df, members)
        print(focus)
        print(ignore)
    elif mode == "arbitrary":
        dfs = arbitrary_mode(df, members)
        print(dfs)
    else:
        df = all_mode(df)
        print(df)
