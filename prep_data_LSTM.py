#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:18:08 2024

@author: Alaina

includes one hot encoding of subject col
currently only performed on _with_saccade

change paths if running on vacc and remove cells
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
    """
    Use the lines with OG_subjects if using onehot. Uncomment the lines that
    use subject instead if not (and comment out the og subjects lines)
    
    maintain even distribution of subjects and runs in train and test
    and account for proportion of MW occurances in each subject to ensure 
    train and test have balanced distribution of MW occurances
    so model generalizes well to new subjects and runs
    """
    # adjusted to ensure the same subjects do not appear in train and test
    
    # calculate MW proportion per subject
    #sub_mw_proportions = dfSamples.groupby(["Subject","run_num"])["is_MW"].mean().reset_index()
    sub_mw_proportions = dfSamples.groupby(["OG_subjects"])["is_MW"].mean().reset_index()
    # rename cols for clarity
    #sub_mw_proportions.columns = ["Subject", "run_num", "mw_proportion"]
    sub_mw_proportions.columns = ["OG_subjects", "mw_proportion"]
    
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
    train_data = pd.merge(dfSamples, train_pairs, on=["OG_subjects"])
    #test_data = pd.merge(dfSamples, test_pairs, on=["Subject", "run_num"])
    test_data = pd.merge(dfSamples, test_pairs, on=["OG_subjects"])
    
    # verify distribution looks good
    train_mw_p = train_data["is_MW"].mean()
    test_mw_p = test_data["is_MW"].mean()
    
    print("Mean of is_MW in train set: ", train_mw_p)
    print("Mean of is_MW in test set: ", test_mw_p)
    return train_data, test_data

# load data 
# vacc path
#file_path = "./all_subjects_interpolated_with_saccade.csv"
# local path
file_path = "E:\\MW_Classifier_Data\\all_subjects_interpolated_with_saccade.csv"
if not os.path.exists(file_path):
    print(f"Error: Data file not found at {file_path}")
print("loading data...")
dfSamples = load_data(file_path)
#dfSamples=pd.read_csv(file_path)

if "Unnamed: 0" in dfSamples.columns:
    dfSamples.drop("Unnamed: 0", axis=1, inplace=True)
    print("Unnamed: 0 dropped.") 
    print("dfSamples Columns: ", dfSamples.columns)
    
# one hot subjects
# preserve sub column for splitting
dfSamples["OG_subjects"] = dfSamples["Subject"]
one_hot_samples = pd.get_dummies(dfSamples, columns=["Subject"], prefix="sub")

pd.set_option('display.max_columns', None) 
print(one_hot_samples.head())
    
# train test split
train_data, test_data = split_data(one_hot_samples)

# print subjects in each to verify
print("train subjects: ", train_data["OG_subjects"].unique())
print("test subjects", test_data["OG_subjects"].unique())

# save train, test - vacc
#train_data.to_csv("./train_with_saccade_onehot_subjects.csv", index=False)
#test_data.to_csv("./test_with_saccade_onehot_subjects.csv", index=False)

# save to train, test - local
train_data.to_csv("C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\train_with_saccade_onehot_subs.csv", index=False)
test_data.to_csv("C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\test_with_saccade_onehot_subs.csv", index=False)

#%%

# cast one hot cols to int
train_one_hot_cols = train_data.filter(like="sub_").columns
print(train_one_hot_cols)
train_data[train_one_hot_cols] = train_data[train_one_hot_cols].astype(int)

test_one_hot_cols = test_data.filter(like="sub_").columns
print(test_one_hot_cols)
test_data[test_one_hot_cols] = test_data[test_one_hot_cols].astype(int)



# verify casting worked
print(train_data.dtypes)
print(test_data.dtypes)

#%%
# check one hot values- should only see 0s for one hot cols for subjects that
# aren't present in a set
"""
train subjects:  [10014 10052 10059 10073 10080 10081 10084 10085 10092 10094 10100 10103
 10115 10121 10125]
test subjects [10089 10110 10111 10117]
"""
print("train")
for col in train_one_hot_cols:
    print(f"{col}: {train_data[col].unique()}")
    
print("test")
for col in test_one_hot_cols:
    print(f"{col}: {test_data[col].unique()}")
    
#%%
# save csvs - local and netfiles
# local
print("saving train- local")
train_data.to_csv("C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\train_with_saccade_onehot_subs.csv", index=False)

print("saving test- local")
test_data.to_csv("C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\test_with_saccade_onehot_subs.csv", index=False)

print("saving train-netfiles")
train_data.to_csv("Z:\\Mindless Reading\\neuralnet_classifier\\train_with_saccade_onehot_subs.csv", index=False)

print("saving test- netfiles")
test_data.to_csv("Z:\\Mindless Reading\\neuralnet_classifier\\test_with_saccade_onehot_subs.csv", index=False)
