# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 07:17:02 2024

@author: abirn

investigating subject mw distribution, removing instances of not mw to reduce
imbalance
"""

import pandas as pd
from matplotlib import pyplot as plt

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

def mw_over_time(dfSamples, cleaned=False):
    """
    Generates a line plot of the number of times that mind wandering was reported
    as a function of time for each subject and run. The line plots are saved to the
    plots folder within the current working directory.

    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.

    Returns
    -------
    None
    """
    # get MW occurances for each timestep for each sub run pair
    MW_over_time_subrun = dfSamples.groupby(["Subject", "run_num", "tSample_normalized"])["is_MW"].sum()
    
    # get unique subjects
    subjects = dfSamples["Subject"].unique()
    # plot
    for subject in subjects:
        # get data for this subject
        subject_data = MW_over_time_subrun.loc[subject]
        # make fig for this sub
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"MW Occurances Over Time: Subject {subject}")
        # subplot for each run
        for run in range(1,6):
            if run not in subject_data.index.get_level_values("run_num"):
                print(f"Run {run} for Subject {subject} is missing after cleaning. Skipping.")
                continue
            run_data = subject_data.loc[run]
            plt.subplot(3,2,run)
            plt.plot(run_data.index, run_data.values)
            plt.xlabel("Normalized Time (ms from run start)")
            plt.ylabel("MW Frequency")
            plt.title("Run {run}")
            plt.tight_layout()
        
        # save when done
        if cleaned == False:
            plt.savefig(f"./Plots/MW_freq_over_time_s{subject}.png")
        else:
            plt.savefig(f"./Plots/MW_freq_over_time_s{subject}_nonmw_removed.png")
        plt.close()
        
def remove_latent(dfSamples, buffer=4000):
    """
    Add a buffer around mw periods to signify that these not mw samples 
    shouldn't be dropped bc they are too close to mw periods. Do this per 
    subject run pair to avoid making global assumptions. Then get rid of the
    periods that aren't mw or within buffer.
    """
    grouped = dfSamples.groupby(["Subject", "run_num"])
    cleaned_groups = []
    for (subject, run), group in grouped:
        print(f"Processing subject {subject} run {run}")
        group= group.copy()
        # keep col for flag within buffer
        group["keep"] = 0# initialize to 0
        mw_indices = group[group["is_MW"] == 1].index 
        # loop through mw indices to add flag to samples within buffer of mw
        for idx in mw_indices:
            start = group.loc[idx, "tSample_normalized"] - buffer
            end = group.loc[idx, "tSample_normalized"] + buffer

            # set keep flag 
            group.loc[(group["tSample_normalized"] >= start) & 
                      (group["tSample_normalized"] <= end), "keep"] =1
            
        # identify the unflagged, nonmw periods
        group["to_remove"] = (group["is_MW"] ==0) & (group["keep"] == 0)
        group = group[~group["to_remove"]].copy()
        cleaned_groups.append(group.reset_index(drop=True))
        
    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    return cleaned_df
        

# import data
train = load_data("./train.csv")

# plot is mw over time for each subject run pair (one plot for each subject, subplot for each run)
#mw_over_time(train)

cleaned_df = remove_latent(train)
# plot again
mw_over_time(cleaned_df, cleaned=True)
# get class imbalance
train_mw_count = (cleaned_df["is_MW"] == 1).sum()
train_non_mw_count = (cleaned_df["is_MW"] == 0).sum()
# .1 before cleaning
print(f"Cleaned Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")
print(f"Lenth before cleaning: {len(train)}")
print(f"length after cleaning: {len(cleaned_df)}")

print("saving to csv (train_balanced)")
cleaned_df.to_csv("./train_balanced", index=False)
    