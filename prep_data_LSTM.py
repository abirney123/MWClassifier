#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:18:08 2024

@author: Alaina
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(file_path, chunksize = 500000):
    # load data in chunks
    chunks = []
    chunk_count = 0
    total_rows = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunks.append(chunk)
        chunk_count += 1
        total_rows += len(chunk)
        # print progress
        print(f"Processed chunk {chunk_count}, total rows processed: {total_rows}")
    # concat chunks into df
    dfSamples = pd.concat(chunks,ignore_index=True)
    # drop unnamed 0 if present
    if "Unnamed: 0" in dfSamples.columns:
        dfSamples.drop(labels=["Unnamed: 0"], axis=1, inplace=True)
    return dfSamples

def split_data(dfSamples):
    # maintain even distribution of subjects and runs in train and test
    # and account for proportion of MW occurances in each subject to ensure 
    # train and test have balanced distribution of MW occurances
    # so model generalizes well to new subjects and runs
    
    # adjusted to ensure the same subjects do not appear in train and test
    
    # calculate MW proportion per subject
    #sub_mw_proportions = dfSamples.groupby(["Subject","run_num"])["is_MW"].mean().reset_index()
    sub_mw_proportions = dfSamples.groupby(["Subject"])["is_MW"].mean().reset_index()
    # rename cols for clarity
    #sub_mw_proportions.columns = ["Subject", "run_num", "mw_proportion"]
    sub_mw_proportions.columns = ["Subject", "mw_proportion"]
    
    # use pd cut to separate mw proportion into bins to represent low medium and high mw occurances
    sub_mw_proportions["mw_bin"] = pd.cut(sub_mw_proportions["mw_proportion"], bins=3, labels=["low", "medium", "high"])
    
    # split sub run pairs into train and test 
    # shuffle is true here because just shuffling the sub run pairs, not the 
    # time series data
    # stratified split by mw proportion 
    train_pairs, test_pairs = train_test_split(sub_mw_proportions,
                                               test_size = .2, random_state=42,
                                               stratify=sub_mw_proportions["mw_bin"])
    
    
    # merge back to train and test to get full sets
    #train_data = pd.merge(dfSamples, train_pairs, on=["Subject", "run_num"])
    train_data = pd.merge(dfSamples, train_pairs, on=["Subject"])
    #test_data = pd.merge(dfSamples, test_pairs, on=["Subject", "run_num"])
    test_data = pd.merge(dfSamples, test_pairs, on=["Subject"])
    
    # verify distribution looks good
    train_mw_p = train_data["is_MW"].mean()
    test_mw_p = test_data["is_MW"].mean()
    
    print("Mean of is_MW in train set: ", train_mw_p)
    print("Mean of is_MW in test set: ", test_mw_p)
    return train_data, test_data

# load data 
# vacc path
file_path = "./all_subjects_interpolated.csv"
# local path
#file_path = "/Volumes/brainlab/Mindless Reading/neuralnet_classifier/all_subjects_interpolated.csv"
if not os.path.exists(file_path):
    print(f"Error: Data file not found at {file_path}")
print("loading data...")
#dfSamples = load_data(file_path)
dfSamples=pd.read_csv(file_path)

if "Unnamed: 0" in dfSamples.columns:
    dfSamples.drop("Unnamed: 0", axis=1, inplace=True)
    print("Unnamed: 0 dropped.") 
    print("dfSamples Columns: ", dfSamples.columns)
    
# train test split
train_data, test_data = split_data(dfSamples)

# print subjects in each to verify
print("train subjects: ", train_data["Subject"].unique())
print("test subjects", test_data["Subject"].unique())


# save train, test
train_data.to_csv("./train.csv", index=False)
test_data.to_csv("./test.csv", index=False)