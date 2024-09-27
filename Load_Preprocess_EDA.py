#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:16:27 2024

@author: Alaina

Load, Preprocess, and EDA for MW Classifier

Blinks and samples with blinks interpolated have been saved as CSVs on Lexar drive
"""

#%% imports

import numpy as np
import pandas as pd
import seaborn as sns
from parse_EyeLink_asc import ParseEyeLinkAsc as parse_asc # from GBL github
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from extract_raw_samples import extract_raw as extract


#%% functions

def load_raw_samples(data_path, subjects): 
    """
    Load data for all subjects specified in subjects list and concatenate
    into a single dataframe.

    Parameters
    ----------
    data_path : Str
        The path to the folder holding raw datafiles.
    subjects : List of int
        The subjects to load data for

    Returns
    -------
    all_s_df : Dataframe
        Raw data for all subjects

    """
    # initialize list to store all df 
    all_s_dfs = []
    
    # loop through subjects to load all data files
    for subject_num in subjects:
        # get subject df
        print(f"Loading subject {subject_num}")
        # change file path depending on if you want to load interpolated data files or not
        #df = pd.read_csv(f"{data_path}{subject_num}_raw_interpolated.csv", dtype={11:str})
        df = pd.read_csv(f"{data_path}{subject_num}_raw.csv", dtype={10:str})
        # add subject id column to ensure even distribution when shuffling later for model
        df["Subject"] = f"{subject_num}"
        # add subject df to df list
        all_s_dfs.append(df)
        
    
    print("num subjects: ", len(all_s_dfs))
    
    for df in all_s_dfs:
        print(df.info())
    
    # combine data for all subjects
    all_s_df = pd.concat(all_s_dfs,ignore_index=True)
    
    return all_s_df

def load_blink_data(subjects, data_path):
    """
    Load blink data for all subjects specified in subjects list, concatenate
    into a single dataframe, and save that dataframe as a CSV.

    Parameters
    ----------
    data_path : Str
        The path to the folder holding blink datafiles.
    subjects : List of int
        The subjects to load data for.

    Returns
    -------
    None.
    """
    # generate blink df 
    # will create one for each subject and run
    # append each dfBlink (with additional col for subject and run) to one large df blink for all subs and runs

    all_blinks_df = []
    for subject_num in subjects:
        for run_num in range(1,6):
            subject_num = str(subject_num)
            elFilename = f"{data_path}ASC_Files/{subject_num}/s{subject_num[-3:]}_r{run_num}.asc"
            dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = parse_asc(elFilename)
            dfBlink["Subject"] = subject_num
            dfBlink["run_num"] = f"{run_num}"
            all_blinks_df.append(dfBlink)
            
            
    # combine data for all blinks
    all_blinks_df = pd.concat(all_blinks_df, ignore_index=True)
    
    # save blink data to csv so we don't need to run Get blink df cell again
    all_blinks_df.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_blinks.csv", index=False)
    
def print_info(dfSamples):
    """
    Print some exploratory information about the datafile holding raw samples.

    Parameters
    ----------
    dfSamples : Dataframe
        Raw samples for every subject.

    Returns
    -------
    None.

    """
    pd.set_option('display.max_columns', None)
    print(dfSamples.describe())
    
    #print("num samples: ",len(dfSamples))
    
    # get distinct values for is_MW to check that it is only 0 or 1
    is_MW_vals = dfSamples["is_MW"].unique()
    print("is_MW values: ", is_MW_vals)
    
    # get num na for each col
    print("Number of Missing Values for Each Column: ")
    print(dfSamples.isna().sum())
    
def interpolate_pupil_over_blinks(dfSamples, dfBlink, subject_num, run):
    """
    Interpolate left and right pupil sizes over blink periods. Modifies the
    dataframe of samples in place to change pupil dilation values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Inputs:
        - dfSamples: A dataframe containing samples for all subjects and all runs
        - dfBlink: A dataframe containing information about the eye in which a 
        blink occured and the time that that blink occured
        - subject_num: String
        The subject number for the subject whos data is currently being interpolated
        - run: Int
        The run number for the data that is currently being interpolated
        
    Returns:
        None
    """
    # extracted from reading_analysis.py (author: HS)
    # adjusted to work on dfs for all subjects and all runs and interpolate left and right pupil by AB
    # https://github.com/GlassBrainLab/MindlessReadingAnalysis/blob/main/EyeAnalysisCode/reading_analysis.py
    # interpolate the pupil size during the blink duration
    # http://dx.doi.org/10.6084/m9.figshare.688002
    
    """
    s10014 has tSample 898100, then jumps to 934682 which is why matches aren't being found.
    this is true regardless of when rows with missing run num are dropped
    added valid_interpolation flag to skip blinks that happen at times like this
    
    Run interpolation on sample csv before extracting raw samples!
    Do for coordinates too 
    
    """
    print(f"Interpolating for subject {subject_num}, run {run}")
    # get subset for this subject and run
    # sample data
    this_sub_run_samples = dfSamples
    # blink data
    this_sub_run_blinks = dfBlink[(dfBlink["Subject"] == subject_num) & 
                                (dfBlink["run_num"] == run)]
    # get the time of every sample
    sample_time = this_sub_run_samples['tSample'].to_numpy()
    #print(sample_time[10999:11999])
    LPupil = np.array(this_sub_run_samples['LPupil'])
    RPupil = np.array(this_sub_run_samples['RPupil'])
    # declare blink offset: 50ms added to the start and end of a blink
    blink_off = 50
    
    # declare variables to store blink start information to merge blinks if 
    # they are too close 
    update_start = True
    # iterate throu each row of blink dataframe
    print("num blinks for this subject and run: ", len(this_sub_run_blinks))
    print("num samples for this subject and run: ", len(this_sub_run_samples))
    for index in np.arange(len(this_sub_run_blinks)):
        # reset flag
        valid_interpolation = True
        row = this_sub_run_blinks.iloc[index]
        # get the start and end time
        cb_start = row['tStart'] - blink_off
        b_end = row['tEnd'] + blink_off
        # update b_start if necessary
        if update_start:
            b_start = cb_start
            
        if index+1 < len(this_sub_run_blinks):
            # get next blink sample
            nrow = this_sub_run_blinks.iloc[index+1]
            nb_start = nrow['tStart'] - blink_off
            
            # check if two blinks are too close to each other
            if b_end >= nb_start:
                # merge two blinks into a longer one
                update_start = False
                continue
            
        # get the blink duration
        blink_dur = b_end - b_start
        update_start = True
        # define time for interpolation
        # t1 ==blink dur== t2 ==blink dur== t3 ==blink dur== t4
        t2 = b_start
        t3 = b_end
        t1 = t2 - blink_dur
        t4 = t3 + blink_dur
        
        
        if (t1 > sample_time[0]) and (t4 < sample_time[-1]):
            #print(f"t1: {t1}, t4: {t4}")
            #print(f"sample_time[0]: {sample_time[0]}, sample_time[-1]: {sample_time[-1]}")
            index_t1 = np.abs(sample_time - t1).argmin()
            index_t4 = np.abs(sample_time - t4).argmin()
            
            # Print sample_time values around t1 and t4
            #print(f"Index of closest time to t1: {index_t1}, value: {sample_time[index_t1]}")
            #print(f"Index of closest time to t4: {index_t4}, value: {sample_time[index_t4]}")

            # select time and pupil size for interpolation
            x = [t1, t2, t3, t4]
            # interpolation for Lpupil
            if row["eye"] == "L":
                y_ind_L = []
                for t in x:
                    if t in sample_time:
                        y_ind_L.append(np.where(sample_time==t)[0][0])
                    else:
                        print(f"No exact match found for time {t}. Skipping this blink.")
                        valid_interpolation = False
                        break 
                    #closest_index = np.argmin(np.abs(sample_time - t))
                    #y_ind_L.append(closest_index)
                if valid_interpolation:
                    y_L = LPupil[y_ind_L]
                    # generate the spl model using the time and pupil size
                    spl_L = CubicSpline(x, y_L)
            # interpolation for Rpupil
            if row["eye"] == "R":
                y_ind_R = []
                for t in x:
                    if t in sample_time:
                    #closest_index = np.argmin(np.abs(sample_time - t))
                        y_ind_R.append(np.where(sample_time==t)[0][0])
                    else:
                        print(f"No exact match found for time {t}. Skipping this blink.")
                        valid_interpolation = False
                        break 
                    #y_ind_R.append(closest_index)
                if valid_interpolation:
                    y_R = RPupil[y_ind_R]
                    # generate the spl model using the time and pupil size
                    spl_R = CubicSpline(x, y_R)
            if valid_interpolation:
                # generate mask for blink duration
                mask = (sample_time > t2) & (sample_time < t3)
                # Print values before the update
                indices_to_update = this_sub_run_samples.index[mask]
                #print("Values before update (LPupil):")
                #print(dfSamples.loc[indices_to_update, "LPupil"])
                x = sample_time[mask]
                # use spl model to interpolate pupil size for blink duration
                # do for each pupil
                if row["eye"] == "L":
                    interp_Lpupil = spl_L(x)
                if row["eye"] == "R":
                    interp_Rpupil = spl_R(x)
                
                # update the df for this subject and run
                # Update dfSamples directly within the loop
                if row["eye"] == "L":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "LPupil"] = interp_Lpupil
                    # Print values after the update
                    #print("Values after update (LPupil):")
                    #print(dfSamples.loc[indices_to_update, "LPupil"])
                if row["eye"] == "R":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "RPupil"] = interp_Rpupil

                    
    dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject}/s{subject_str[-3:]}_r{run}_Sample_Interpolated.csv")
    
def interpolate_coordinates_over_blinks(dfSamples, dfBlink, subject_num, run):
    """
    Interpolate left and right x and y eye coordinates over blink periods. Modifies the
    dataframe of samples in place to change coordinate values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Inputs:
        - dfSamples: A dataframe containing samples for all subjects and all runs
        - dfBlink: A dataframe containing information about the eye in which a 
        blink occured and the time that that blink occured
        
    Returns:
        None.
    """
    # extracted from reading_analysis.py (author: HS)
    # adjusted to work on dfs for all subjects and all runs and interpolate left and right pupil by AB
    # https://github.com/GlassBrainLab/MindlessReadingAnalysis/blob/main/EyeAnalysisCode/reading_analysis.py
    # interpolate the pupil size during the blink duration
    # http://dx.doi.org/10.6084/m9.figshare.688002
    
    """
    s10014 has tSample 898100, then jumps to 934682 which is why matches aren't being found.
    this is true regardless of when rows with missing run num are dropped
    added valid_interpolation flag to skip blinks that happen at times like this
    
    Run interpolation on sample csv before extracting raw samples!
    Do for coordinates too 
    
    """
    print(f"Interpolating for subject {subject_num}, run {run}")
    # get subset for this subject and run
    # sample data
    this_sub_run_samples = dfSamples
    # blink data
    this_sub_run_blinks = dfBlink[(dfBlink["Subject"] == subject_num) & 
                                (dfBlink["run_num"] == run)]
    # get the time of every sample
    sample_time = this_sub_run_samples['tSample'].to_numpy()
    #print(sample_time[10999:11999])
    LX = np.array(this_sub_run_samples['LX'])
    LY = np.array(this_sub_run_samples['LY'])
    RX = np.array(this_sub_run_samples['RX'])
    RY = np.array(this_sub_run_samples['RY'])
    # declare blink offset: 50ms added to the start and end of a blink
    blink_off = 50
    
    # declare variables to store blink start information to merge blinks if 
    # they are too close 
    update_start = True
    # iterate throu each row of blink dataframe
    print("num blinks for this subject and run: ", len(this_sub_run_blinks))
    print("num samples for this subject and run: ", len(this_sub_run_samples))
    for index in np.arange(len(this_sub_run_blinks)):
        # reset flag
        valid_interpolation = True
        row = this_sub_run_blinks.iloc[index]
        # get the start and end time
        cb_start = row['tStart'] - blink_off
        b_end = row['tEnd'] + blink_off
        # update b_start if necessary
        if update_start:
            b_start = cb_start
            
        if index+1 < len(this_sub_run_blinks):
            # get next blink sample
            nrow = this_sub_run_blinks.iloc[index+1]
            nb_start = nrow['tStart'] - blink_off
            
            # check if two blinks are too close to each other
            if b_end >= nb_start:
                # merge two blinks into a longer one
                update_start = False
                continue
            
        # get the blink duration
        blink_dur = b_end - b_start
        update_start = True
        # define time for interpolation
        # t1 ==blink dur== t2 ==blink dur== t3 ==blink dur== t4
        t2 = b_start
        t3 = b_end
        t1 = t2 - blink_dur
        t4 = t3 + blink_dur
        
        
        if (t1 > sample_time[0]) and (t4 < sample_time[-1]):
            #print(f"t1: {t1}, t4: {t4}")
            #print(f"sample_time[0]: {sample_time[0]}, sample_time[-1]: {sample_time[-1]}")
            index_t1 = np.abs(sample_time - t1).argmin()
            index_t4 = np.abs(sample_time - t4).argmin()
            
            # Print sample_time values around t1 and t4
            #print(f"Index of closest time to t1: {index_t1}, value: {sample_time[index_t1]}")
            #print(f"Index of closest time to t4: {index_t4}, value: {sample_time[index_t4]}")
    
            # select time and pupil size for interpolation
            x = [t1, t2, t3, t4]
            # interpolation for LX
            if row["eye"] == "L":
                y_ind_LX = []
                y_ind_LY = []
                for t in x:
                    if t in sample_time:
                        y_ind_LX.append(np.where(sample_time==t)[0][0])
                        y_ind_LY.append(np.where(sample_time==t)[0][0])
                    else:
                        print(f"No exact match found for time {t}. Skipping this blink.")
                        valid_interpolation = False
                        break 
                    #closest_index = np.argmin(np.abs(sample_time - t))
                    #y_ind_L.append(closest_index)
                if valid_interpolation and np.all(np.isfinite(LX[y_ind_LX])) and np.all(np.isfinite(LY[y_ind_LY])):
                    y_LX = LX[y_ind_LX]
                    y_LY = LY[y_ind_LY]
                    # generate the spl model using the time and pupil size
                    spl_LX = CubicSpline(x, y_LX)
                    spl_LY = CubicSpline(x, y_LY)
                else:
                    print(f"Non-finite values found for Left Eye. Skipping blink at index {index}.")
                    valid_interpolation = False
            # interpolation for Rpupil
            if row["eye"] == "R":
                y_ind_RX = []
                y_ind_RY = []
                for t in x:
                    if t in sample_time:
                    #closest_index = np.argmin(np.abs(sample_time - t))
                        y_ind_RX.append(np.where(sample_time==t)[0][0])
                        y_ind_RY.append(np.where(sample_time==t)[0][0])
                    else:
                        print(f"No exact match found for time {t}. Skipping this blink.")
                        valid_interpolation = False
                        break 
                    #y_ind_R.append(closest_index)
                if valid_interpolation and np.all(np.isfinite(RX[y_ind_RX])) and np.all(np.isfinite(RY[y_ind_RY])):
                    y_RX = RX[y_ind_RX]
                    y_RY = RY[y_ind_RY]
                    # generate the spl model using the time and pupil size
                    spl_RX = CubicSpline(x, y_RX)
                    spl_RY = CubicSpline(x, y_RY)
            if valid_interpolation and np.all(np.isfinite(RX[y_ind_RX])) and np.all(np.isfinite(RY[y_ind_RY])):
                # generate mask for blink duration
                mask = (sample_time > t2) & (sample_time < t3)
                # Print values before the update
                indices_to_update = this_sub_run_samples.index[mask]
                #print("Values before update (LPupil):")
                #print(dfSamples.loc[indices_to_update, "LPupil"])
                x = sample_time[mask]
                # use spl model to interpolate pupil size for blink duration
                # do for each pupil
                if row["eye"] == "L":
                    interp_LX = spl_LX(x)
                    interp_LY = spl_LY(x)
                if row["eye"] == "R":
                    interp_RX = spl_RX(x)
                    interp_RY = spl_RY(x)
            else:
                print(f"Non-finite values found for Right Eye. Skipping blink at index {index}.")
                valid_interpolation = False
            if valid_interpolation:  
                # update the df for this subject and run
                # Update dfSamples directly within the loop
                if row["eye"] == "L":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "LX"] = interp_LX
                    dfSamples.loc[this_sub_run_samples.index[mask], "LY"] = interp_LY
                    # Print values after the update
                    #print("Values after update (LPupil):")
                    #print(dfSamples.loc[indices_to_update, "LPupil"])
                if row["eye"] == "R":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "RX"] = interp_RX
                    dfSamples.loc[this_sub_run_samples.index[mask], "RY"] = interp_RY

                
    dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject}/s{subject_str[-3:]}_r{run}_Sample_Interpolated.csv")

def preprocess(dfSamples):
    """
    Performs preprocessing on the dataframe of raw samples for each subject. 
    Drops rows with missing run num and creates the tSample_normalized column,
    which is a column of normalized time such that each run begins at time 0.

    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of raw samples for each subject.

    Returns
    -------
    dfSamples: Dataframe
        The dataframe of raw samples for each subject.

    """
    # drop rows with missing run number (tSample norm wont be calculated correctly and they aren't relevant)
    dfSamples.dropna(subset="run_num", inplace=True)
    print("Rows with missing run num have been dropped.")
    
    # for each subject and run, min tSample is start of study - normalize tSample
    dfSamples['tSample_normalized'] = dfSamples.groupby(['Subject',
                                                       'run_num'])['tSample'].transform(lambda x: x - x.min())
    print("tSample_normalized has been created.")
    return dfSamples

def correlation_heatmap(dfSamples):
    """
    Generates a heatmap of the Pearson Correlation Coefficients for columns
    in the dataframe of samples for each subject. The heatmap is saved to the 
    plots folder within the current working directory.

    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.

    Returns
    -------
    None
    """
    # get rid of subject and sample id
    data = dfSamples[["tSample", "tSample_normalized", "RX", "RY", "RPupil", "LX", "LY", "LPupil", "page_num", "run_num", "is_MW"]]
    
    corr_matrix = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin = -1, vmax = 1)
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.show()
    
    plt.savefig("plots/correlation_heatmap.png")


def mw_freq_by_subject(dfSamples):
    """
    Generates a bar plot of the number of times that mind-wandering was reported
    for each subject and saves that bar plot to the plots folder within the current 
    working directory.

    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.

    Returns
    -------
    None
    """
    # count 1s in is_MW for each distinct value in subject col
    is_MW_count = dfSamples.groupby("Subject")["is_MW"].sum()
    
    plt.figure(figsize=(10,8))
    # plot MW frequency for each subject
    plt.bar(is_MW_count.index, is_MW_count.values)
    
    plt.xlabel("Subject")
    plt.xticks(rotation="vertical")
    plt.ylabel("Number of Samples where MW Occurred")
    plt.title('Occurances of MW by Subject')
    plt.tight_layout()
    plt.show()
    
    plt.savefig("plots/MW_freq_per_subject.png")


def mw_over_time(dfSamples):
    """
    Generates a line plot of the number of times that mind wandering was reported
    as a function of time. The number of times that mind wandering was reported
    is aggregated over subjects and runs. The line plot is also saved to the
    plots folder within the current working directory.

    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.

    Returns
    -------
    None
    """
    # get MW occurances for each tSample normalized
    MW_over_time = dfSamples.groupby("tSample_normalized")["is_MW"].sum()
    
    # plot
    
    plt.figure(figsize=(10, 6))
    
    # Plot mind-wandering count over time
    plt.plot(MW_over_time.index, MW_over_time.values)
    
    # Add labels and title
    plt.xlabel("Normalized Time (ms from run start)")
    plt.ylabel("MW Frequency")
    plt.title("MW Occurrances Over Time")
    
    plt.show()
    
    plt.savefig("plots/MW_freq_over_time.png")


def pupil_over_time(dfSamples):
    """
    Generates a line plot of the pupil dilation over time, averaged over subjects
    and runs. The line plot is then saved to the plots folder within the current
    working directory.

    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.

    Returns
    -------
    None
    """
    
    LPupil_over_time = dfSamples.groupby("tSample_normalized")["LPupil"].mean()
    
    RPupil_over_time = dfSamples.groupby("tSample_normalized")["RPupil"].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(LPupil_over_time.index, LPupil_over_time.values, label="Left Pupil")
    plt.plot(RPupil_over_time.index, RPupil_over_time.values, alpha=.4, label="Right Pupil")
    plt.xlabel("Normalized Time (ms from run start)")
    plt.ylabel("Dilation of Pupil (averaged over subjects and runs)")
    plt.title("Pupil Dilation Over Time")
    plt.legend()
    
    plt.savefig("plots/dilation_over_time_avg_sub_run.png")



def pupil_subject_run(dfSamples, subjects):
    """
    Generates one plot for each subject with subplots for each run. Each subplot
    shows the pupil dilation over time for that subject and run. Each plot is 
    saved to the plots folder within the current working directory.
    
    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.
    subjects : List of int
        The subjects to plot data for.

    Returns
    -------
    None.

    """
    for subject_num in subjects:
        fig, axes = plt.subplots(2,3, figsize=(15,10))
        fig.suptitle(f"Pupil Dilation for Subject {subject_num}")
        axes = axes.flatten()
        for run_num in range(1,6): # 5 runs for each subject
            this_sub_run_df = dfSamples[(dfSamples["Subject"] == str(subject_num)) &
                                       (dfSamples["run_num"] == float(run_num))]
            # plot pupil dilation for this subject and run
            axes[run_num-1].plot(this_sub_run_df["tSample_normalized"],
                               this_sub_run_df["LPupil"], label="Left Pupil")
            axes[run_num-1].plot(this_sub_run_df["tSample_normalized"],
                               this_sub_run_df["RPupil"], alpha = .4, label="Right Pupil")
            # set labels
            axes[run_num-1].set_title(f"Run {run_num}")
            axes[run_num-1].set_xlabel("Normalized Time (ms from run start)")
            axes[run_num-1].set_ylabel("Pupil Dilation")
            # rotate x ticks
            axes[run_num-1].tick_params(axis="x", rotation=45)
            
        # get rid of the empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.savefig(f"plots/Subject{subject_num}_dilation_over_time.png")
        plt.close(fig)
        
def normalize_pupil(dfSamples):
    """
    Perform within-subject normalization for Lpupil and Rpupil by dividing each 
    subjects Lpupil and Rpupil values by the mean of each.
    
    Parameters
    ----------
    dfSamples: Dataframe
        The dataframe of samples for each subject.
    
    Returns
    -------
    dfSamples: Dataframe
        The dataframe of samples for each subject.
    """
    # Normalize LPupil for each subject
    dfSamples["LPupil_normalized"] = dfSamples.groupby("Subject")["LPupil"].transform(lambda x: x / x.mean())
    
    # Normalize RPupil for each subject
    dfSamples["RPupil_normalized"] = dfSamples.groupby("Subject")["RPupil"].transform(lambda x: x / x.mean())
    return dfSamples


def eye_coordinates(dfSamples, subjects):
    """
    Generates one plot for each subject with subplots for each run. Each subplot
    shows the left and right eye coordinates for that subject and run. Each plot is 
    saved to the plots folder within the current working directory.
    
    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.
    subjects : List of int
        The subjects to plot data for.

    Returns
    -------
    None.
    """
    # one plot for each subject with subplots for each run
    for subject_num in subjects:
        fig, axes = plt.subplots(2,3, figsize=(15,10))
        fig.suptitle(f"Eye Coordinates for Subject {subject_num}")
        axes = axes.flatten()
        for run_num in range(1,6): # 5 runs for each subject
            # get subset of dataframe for this subject and run
         # get subset of dataframe for this subject and run
             this_sub_run_df = dfSamples[(dfSamples["Subject"] == str(subject_num)) &
                                        (dfSamples["run_num"] == float(run_num))]
             # plot pupil dilation for this subject and run
             axes[run_num-1].plot(this_sub_run_df["LX"],
                                this_sub_run_df["LY"], label="Left Eye", color="red")
             axes[run_num-1].plot(this_sub_run_df["RX"],
                                this_sub_run_df["RY"], alpha = .6, label="Right Eye", color="green")
             # set labels
             axes[run_num-1].set_title(f"Run {run_num}")
             axes[run_num-1].set_xlabel("X position")
             axes[run_num-1].set_ylabel("Y position")
         
         # get rid of the empty subplot
        fig.delaxes(axes[5])
         
        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.savefig(f"plots/Subject{subject_num}_coordinates.png")
        plt.close(fig)
    
#%% Gaze coordinates over time (averaged over subjects and runs)
# just to check for artifacts

#%% Interpolate Pupil
# run interpolation on samples csv for each subject and run
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
data_path = "/Volumes/Lexar/" 
dfBlink = pd.read_csv(f"{data_path}MW_Classifier_Data/all_s_blinks.csv")
for subject in subjects:
    for run in range(1,6):
        subject_str = str(subject)
        dfSamples = pd.read_csv(f"{data_path}/CSV_Samples/{subject}/s{subject_str[-3:]}_r{run}_Sample.csv")
        interpolate_pupil_over_blinks(dfSamples, dfBlink, subject, run)
        
#%% Interpolate Coordinates - have not done yet!

for subject in subjects:
    for run in range(1,6):
        subject_str = str(subject)
        dfSamples = pd.read_csv(f"{data_path}/CSV_Samples/{subject}/s{subject_str[-3:]}_r{run}_Sample.csv")
        interpolate_coordinates_over_blinks(dfSamples, dfBlink, subject, run)

#%% extract

# create raw samples from those interpolated csvs (extract_raw_samples)
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]

for subject in subjects:
    folder_path = f"/Volumes/Lexar/CSV_Samples/{subject}"
    extract(subject, folder_path)

#%% Preprocess

subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
data_path = "/Volumes/Lexar/MW_Classifier_Data/"
dfSamples = load_raw_samples(data_path, subjects)
dfSamples = preprocess(dfSamples)

#%%
print(dfSamples["run_num"].unique())
print(dfSamples["Subject"].unique())
print(dfSamples.columns)

#%%
print(dfSamples["Subject"])
print(dfSamples["run_num"])
#%% Plot
# missing data for subjects and runs or not finding it with existing code
# likely bc run num is float instead of int and subject num is string
# adjust plot code as needed
#correlation_heatmap(dfSamples)
#mw_freq_by_subject(dfSamples)
#mw_over_time(dfSamples)
#pupil_over_time(dfSamples)
pupil_subject_run(dfSamples, subjects)
eye_coordinates(dfSamples, subjects)

#%% some checks
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
for subject_num in subjects:
    for run_num in range(1,6): # 5 runs for each subject
        this_sub_run_df = dfSamples[(dfSamples["Subject"] == str(subject_num)) &
                                    (dfSamples["run_num"] == float(run_num))]
        if this_sub_run_df.empty:
            print(f"No data for Subject {subject_num}, Run {run_num}")
            continue
        else:
            print("not empty for Subject {subject_num}, Run {run_num}")
    
    
print(this_sub_run_df[["tSample_normalized", "LPupil", "RPupil"]].head())

#%%
print(dfSamples.isna().sum())

#%%
dfSamples.to_CSV("/Volumes/Lexar/MW_Classifier_Data/Samples_all_s_interpolated_pupil.csv")

#%% old setup (interpolate after getting raw samples)
data_path = "/Volumes/Lexar/MW_Classifier_Data/" 
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]

dfSamples = load_raw_samples(data_path, subjects)
# uncomment the following two lines if you do not have blink data saved as CSV yet
# dont forget to add a line to change datapath back to point to MW_Classifier_Data
# before getting dfBlink from the CSV if you do this
# data_path = "/Volumes/Lexar/ASC_Files/" 
# load_blink_data(subjects, data_path)


# print_info(dfSamples)

dfSamples = preprocess(dfSamples)
dfSamples.to_csv(f"{data_path}all_subjects_data_no_interpolation.csv")

#%% old setup continuted
dfBlink = pd.read_csv(f"{data_path}all_s_blinks.csv")

# dfBlink: subject and run_num are ints
# dfSamples: subject is string, run_num is float - change to ints

dfSamples["Subject"] = dfSamples["Subject"].astype(int)

dfSamples["run_num"] = dfSamples["run_num"].astype(int)

#interpolate pupil over blinks and plot

# uncomment the following two lines if you haven't saved the CSV with interpolated
# pupil data yet, dont forget to comment out the line that reads the interpolated 
# data from the csv too
# dfSamples = interpolate_pupil_over_blinks(dfSamples, dfBlink, subjects)
# dfSamples.to_csv(f"{data_path}/interpolated_samples.csv" )

dfSamples = pd.read_csv(f"{data_path}/interpolated_samples.csv")
# make plots
#correlation_heatmap(dfSamples)
#mw_freq_by_subject(dfSamples)
#mw_over_time(dfSamples)
#pupil_over_time(dfSamples)
pupil_subject_run(dfSamples, subjects)
#eye_coordinates(dfSamples, subjects)

#%% Get subject datatype
data_path = "/Volumes/Lexar/MW_Classifier_Data/" 
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]

dfSamples = pd.read_csv(f"{data_path}all_subjects_data_no_interpolation.csv")
print(dfSamples["Subject"])

# subject is int in dfSamples when not interpolated
#%% Normalize pupils and save CSV again

dfSamples = normalize_pupil(dfSamples)
dfSamples.to_csv(f"{data_path}all_subjects_data_no_interpolation.csv")

#%%
# run fntns cell again before running this
print_info(dfSamples)
lpupil_means = dfSamples.groupby('Subject')['LPupil_normalized'].mean()
rpupil_means = dfSamples.groupby('Subject')['RPupil_normalized'].mean()

print(lpupil_means)
print(rpupil_means)

#%% drop sample id, lpupil, and rpupil then save
dfSamples = dfSamples.drop(["sample_id", "LPupil", "RPupil"],axis=1)
print("cols have been dropped")
dfSamples.to_csv(f"{data_path}all_subjects_data_no_interpolation.csv")