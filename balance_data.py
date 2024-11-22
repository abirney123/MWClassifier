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
            plt.title(f"Run {run}")
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
        # get tSample normalized for each sample where mw occurred for this sub run
        mw_times = group.loc[group["is_MW"] == 1, "tSample_normalized"].values
        # handle cases where there is no mw
        if len(mw_times) == 0:
            # don't include this sub run in the cleaned data
            print(f"Removing subject {subject}, run {run}. No MW present.")
            continue
        
        cleaned_groups.append(group.reset_index(drop=True))
        """
        for removing based on buffer- commented out to just remove runs with no mw
        # if there is some mw, create an indication of rows to keep-
        # keep rows where tSample normalized is within buffer of mw_time
        group["keep"] = group["tSample_normalized"].apply(
            lambda x: ((mw_times - buffer <= x) & (x <= mw_times + buffer)).any()).astype(int)
            
        # retain rows where keep is 1 or mw is true
        group = group[(group["is_MW"] == 1) | (group["keep"] == 1)].copy()
        
        # append to cleaned groups
        cleaned_groups.append(group.reset_index(drop=True))
        """
        
    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    return cleaned_df

def select_mw(dfSamples, buffer=4000):
    """
    Identifies mw periods, then selects those periods plus buffer surrounding them 
    to include in dataset
    """
    # group by subject and run
    grouped = dfSamples.groupby(["Subject", "run_num"])
    selected_rows = []
    # handle runs with no mw (drop)
    for (subject, run), group in grouped:
        print(f"Processing subject {subject} run {run}")
        group= group.copy()
        mw_rows = group[group["is_MW"] == 1]
        # handle cases where there is no mw
        if mw_rows.empty:
            # don't include this sub run in the cleaned data
            print(f"Removing subject {subject}, run {run}. No MW present.")
            continue
        # find mw periods - consecutive instances of mw, get tSample for start and end of each period
        # plus buffer
        mw_intervals = [] # mw periods plus buffer- will be list of tuples
        for time in mw_rows["tSample_normalized"]:
            mw_intervals.append((time-buffer, time+buffer))
            
        # merge intervals when they are too close together
        mw_intervals.sort() # sort by start time
        merged_intervals = []
        current_start, current_end = mw_intervals[0]
        
        for start, end in mw_intervals[1:]:
            if start <= current_end: # if next int starts before this one ends
                # set the end to the max of last end and current end
                current_end = max(current_end, end)
            else:
                # no overlap, append as normal
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        # add last interval
        merged_intervals.append((current_start, current_end))
        
        for start, end in merged_intervals:
            # for each of the defined intervals, grab the corresponding samples
            buffered_rows = group[
                (group["tSample_normalized"] >= start) &
                (group["tSample_normalized"] <= end)]
            # and add them to selected rows
            selected_rows.append(buffered_rows)
    # combine
    cleaned_data = pd.concat(selected_rows, axis=0, ignore_index=True) 
    return cleaned_data

        
        

# import data
train = load_data("./train.csv")

#mw_over_time(train, cleaned=False)
buffer = 10000
cleaned_df = select_mw(train, buffer)

# check new distribution
# get class imbalance
train_mw_count = (cleaned_df["is_MW"] == 1).sum()
train_non_mw_count = (cleaned_df["is_MW"] == 0).sum()
print(f"Cleaned Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")
print(f"Lenth before cleaning: {len(train)}")
print(f"length after cleaning: {len(cleaned_df)}")
# check columns
print("Cleaned Columns")
print(cleaned_df.columns)
# plot
mw_over_time(cleaned_df, cleaned=True)


cleaned_df.to_csv("./train_balanced_newstrat_10k.csv", index=False)

# OLD STRATEGY
"""
# plot is mw over time for each subject run pair (one plot for each subject, subplot for each run)
#mw_over_time(train)
buffer = 10000
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
cleaned_df.to_csv("./train_balanced_no_empty_runs.csv", index=False)
"""
    