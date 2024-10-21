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
from scipy.interpolate import interp1d #CubicSpline
from extract_raw_samples import extract_raw as extract
import os


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
        #df = pd.read_csv(f"{data_path}{subject_num}_raw_interpolated_pupil_coord.csv", dtype={11:str})
        df = pd.read_csv(f"{data_path}{subject_num}_raw.csv")
        #df = pd.read_csv(f"{data_path}{subject_num}_raw.csv", dtype={10:str})
        # add subject id column to ensure even distribution when shuffling later for model
        df["Subject"] = f"{subject_num}"
        # add subject df to df list
        all_s_dfs.append(df)
        
    
    # combine data for all subjects
    all_s_df = pd.concat(all_s_dfs,ignore_index=True)
    
    return all_s_df

def load_blink_data(subjects, data_path):
    """
    load all blink csvs (for each subject and run) and concatenate them into
    a single dataframe with new cols for subject and run
    """
    all_blinks_dfs = []
    for subject_num in subjects:
        for run_num in range(1,6):
            sub_num_str = str(subject_num)
            folders = os.listdir(f"{data_path}s{subject_num}/eye")
            matching_folders = [folder for folder in folders if folder.startswith(f"s{sub_num_str[-3:]}_r{run_num}") & folder.endswith("data")]
            print(matching_folders)
            folder = matching_folders[0]
            filenames = os.listdir(f"{data_path}s{subject_num}/eye/{folder}")
            matching_files = [file for file in filenames if file.startswith(f"s{sub_num_str[-3:]}_r{run_num}") & file.endswith("Blink.csv")]
            print(matching_files)
            file = matching_files[0]
            dfBlink = pd.read_csv(f"{data_path}s{subject_num}/eye/{folder}/{file}")
            dfBlink["Subject"] = subject_num
            dfBlink["run_num"] = f"{run_num}"
            all_blinks_dfs.append(dfBlink)
            
    all_blinks_df = pd.concat(all_blinks_dfs, ignore_index=True)
    #print("Saving to CSV...")
    #all_blinks_df.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_blinks_9_30.csv", index=False)
    return all_blinks_df
    
def load_saccade_data(subjects, data_path):
    """
    load all bsaccades csvs (for each subject and run) and concatenate them into
    a single dataframe with new cols for subject and run
    """
    all_sacc_dfs = []
    for subject_num in subjects:
        for run_num in range(1,6):
            sub_num_str = str(subject_num)
            folders = os.listdir(f"{data_path}s{subject_num}/eye")
            matching_folders = [folder for folder in folders if folder.startswith(f"s{sub_num_str[-3:]}_r{run_num}") & folder.endswith("data")]
            print(matching_folders)
            folder = matching_folders[0]
            filenames = os.listdir(f"{data_path}s{subject_num}/eye/{folder}")
            matching_files = [file for file in filenames if file.startswith(f"s{sub_num_str[-3:]}_r{run_num}") & file.endswith("Saccade.csv")]
            print(matching_files)
            file = matching_files[0]
            dfBlink = pd.read_csv(f"{data_path}s{subject_num}/eye/{folder}/{file}")
            dfBlink["Subject"] = subject_num
            dfBlink["run_num"] = f"{run_num}"
            all_sacc_dfs.append(dfBlink)
            
    all_sacc_df = pd.concat(all_sacc_dfs, ignore_index=True)
    #print("Saving to CSV...")
    #all_sacc_df.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_saccades.csv", index=False)
    return all_sacc_df

def load_blink_dataOLD(subjects, data_path):
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
    
def interpolate_pupil_over_blinksv2(dfSamples, dfBlink, dfSaccades, subject_num, run):
    """
    Interpolate left and right pupil sizes over blink periods. Modifies the
    dataframe of samples in place to change pupil dilation values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Uses saccades as t1 and t4. Contains adjustments recommended through conversation
    with Dr. J. Performs the interpolation over the normalized pupil dilation values.
    
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
    #this_sub_run_blinks = dfBlink[(dfBlink["Subject"] == subject_num) & 
                                #(dfBlink["run_num"] == run)]
    this_sub_run_blinks = dfBlink
    # get the time of every sample
    sample_time = this_sub_run_samples['tSample'].to_numpy()
    #print(sample_time[10999:11999])
    LPupil = np.array(this_sub_run_samples['LPupil_normalized'])
    RPupil = np.array(this_sub_run_samples['RPupil_normalized'])
    # declare blink offset: 50ms added to the start and end of a blink
    blink_off = 0#50
    
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
                print("merging blinks")
                # update blink end time if merging blinks
                b_end = nrow["tEnd"] + blink_off
                # merge two blinks into a longer one
                update_start = False
                continue
            
        update_start = True
        # define time for interpolation
        # t1 ==blink dur== t2 ==blink dur== t3 ==blink dur== t4

        
        # get the blink duration
        blink_dur = b_end - b_start
        update_start = True
        # define time for interpolation
        # t1 ==blink dur== t2 ==blink dur== t3 ==blink dur== t4
        t2 = b_start
        t3 = b_end
        # set t1 to be the end time of the last saccade before the blink
        #get all saccades before this blink
        previous_saccades = dfSaccades[dfSaccades["tEnd"] < b_start]
        # get last saccade before this blink
        last_saccade_tEnd = previous_saccades["tEnd"].max()
        # set t4 to be the start time of the first saccade after the blink
        # get all saccades after this blink
        later_saccades = dfSaccades[dfSaccades["tStart"] > b_end]
        # get the first saccade after this blink
        first_saccade_tStart = later_saccades["tStart"].min()
        t1 = last_saccade_tEnd
        t4 = first_saccade_tStart
        # check for missing vals in t1 or t4 and use buffer as fallback
        if pd.isna(t1):
            print("t1 is na, using blink duration")
            t1 = t2 - blink_dur
        if pd.isna(t4):
            print("t4 is na, using blink duration")
            t4 = t3 + blink_dur
        #print("t1", t1, "t2", t2, "t3", t3, "t4", t4)
        
        
        if (t1 > sample_time[0]) and (t4 < sample_time[-1]):

            # select time and pupil size for interpolation
            #x = [t1, t2, t3, t4]
            x = [t1,t4]
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
                    #spl_L = CubicSpline(x, y_L)
                    lin_L = interp1d(x, y_L)
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
                    #spl_R = CubicSpline(x, y_R)
                    lin_R = interp1d(x, y_R)
            if valid_interpolation:
                # generate mask for blink duration
                #mask = (sample_time > t1) & (sample_time < t4)
                # base mask on closest index instead of exact
                idx_t1 = np.argmin(np.abs(sample_time - t1))
                idx_t4 = np.argmin(np.abs(sample_time - t4))
                mask = (sample_time > sample_time[idx_t1]) & (sample_time < sample_time[idx_t4])
                x = sample_time[mask]
                # sample times align with masked time range
                print(f"masking time range: {t1} to {t4}")
                print(f"Sample times in blink period: {sample_time[mask][0]} to {sample_time[mask][-1]}")
                #print(f"LPupil in blink period: {LPupil[mask]}")
                #print(f"RPupil in blink period: {RPupil[mask]}")
                # use spl model to interpolate pupil size for blink duration
                # do for each pupil
                if row["eye"] == "L":
                    interp_Lpupil = lin_L(x)
                if row["eye"] == "R":
                    interp_Rpupil = lin_R(x)
                
                # update the df for this subject and run
                # Update dfSamples directly within the loop
                if row["eye"] == "L":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "LPupil_normalized"] = interp_Lpupil
                    # Print values after the update
                    #print("Values after update (LPupil):")
                    #print(dfSamples.loc[indices_to_update, "LPupil"])
                if row["eye"] == "R":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "RPupil_normalized"] = interp_Rpupil
                    
                # if these were zeros this would cause the interpolation to fail, but they're not
                #print(f"Pupil dilation at t1: {LPupil[y_ind_L[0]] if row['eye'] == 'L' else RPupil[y_ind_R[0]]}")
                #print(f"Pupil dilation at t4: {LPupil[y_ind_L[1]] if row['eye'] == 'L' else RPupil[y_ind_R[1]]}")

            
    dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_num[-3:]}_r{run}_Sample_Interpolated.csv")
    return dfSamples

def interpolate_pupil_over_blinksv2_nomerge(dfSamples, dfBlink, dfSaccades, subject_num, run):
    """
    Interpolate left and right pupil sizes over blink periods. Modifies the
    dataframe of samples in place to change pupil dilation values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Uses saccades as t1 and t4. Contains adjustments recommended through conversation
    with Dr. J. Performs the interpolation over the normalized pupil dilation values.
    
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
    i = 1
    # get subset for this subject and run
    # sample data
    this_sub_run_samples = dfSamples
    # blink data
    #this_sub_run_blinks = dfBlink[(dfBlink["Subject"] == subject_num) & 
                                #(dfBlink["run_num"] == run)]
    this_sub_run_blinks = dfBlink
    # get the time of every sample
    sample_time = this_sub_run_samples['tSample'].to_numpy()
    #print(sample_time[10999:11999])
    LPupil = np.array(this_sub_run_samples['LPupil_normalized'])
    RPupil = np.array(this_sub_run_samples['RPupil_normalized'])

    # iterate throu each row of blink dataframe
    print("num blinks for this subject and run: ", len(this_sub_run_blinks))
    print("num samples for this subject and run: ", len(this_sub_run_samples))
    for index in np.arange(len(this_sub_run_blinks)):
        # reset flag
        valid_interpolation = True
        row = this_sub_run_blinks.iloc[index]
        # get the start and end time
        b_start = row['tStart'] 
        b_end = row['tEnd']

        
        # get the blink duration
        blink_dur = b_end - b_start


        # set t1 to be the end time of the last saccade before the blink
        #get all saccades before this blink
        previous_saccades = dfSaccades[dfSaccades["tEnd"] < b_start]
        # get last saccade before this blink
        last_saccade_tEnd = previous_saccades["tEnd"].max()
        # set t4 to be the start time of the first saccade after the blink
        # get all saccades after this blink
        later_saccades = dfSaccades[dfSaccades["tStart"] > b_end]
        # get the first saccade after this blink
        first_saccade_tStart = later_saccades["tStart"].min()
        t1 = last_saccade_tEnd
        t4 = first_saccade_tStart
        # check for missing vals in t1 or t4 and use fallback if needed
        if pd.isna(t1):
            print("t1 is na, using blink duration")
            t1 = b_start
        if pd.isna(t4):
            print("t4 is na, using blink duration")
            t4 = b_end
        #print("t1", t1, "t2", t2, "t3", t3, "t4", t4)
        
        
        if (t1 > sample_time[0]) and (t4 < sample_time[-1]):

            # select time and pupil size for interpolation
            #x = [t1, t2, t3, t4]
            x = [t1,t4]
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
                    #spl_L = CubicSpline(x, y_L)
                    lin_L = interp1d(x, y_L)
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
                    #spl_R = CubicSpline(x, y_R)
                    lin_R = interp1d(x, y_R)
            if valid_interpolation:
                # generate mask for blink duration
                mask = (sample_time > t1) & (sample_time < t4)
                x = sample_time[mask]
                # sample times align with masked time range
                if i == 1:
                    print(f"masking time range: {t1} to {t4}")
                    print(f"Sample times in blink period: {sample_time[mask][0]} to {sample_time[mask][-1]}")
                
                i+=1
                #print(f"LPupil in blink period: {LPupil[mask]}")
                #print(f"RPupil in blink period: {RPupil[mask]}")
                # use spl model to interpolate pupil size for blink duration
                # do for each pupil
                if row["eye"] == "L":
                    interp_Lpupil = lin_L(x)
                if row["eye"] == "R":
                    interp_Rpupil = lin_R(x)
                
                # update the df for this subject and run
                # Update dfSamples directly within the loop
                if row["eye"] == "L":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "LPupil_normalized"] = interp_Lpupil
                    # Print values after the update
                    #print("Values after update (LPupil):")
                    #print(dfSamples.loc[indices_to_update, "LPupil"])
                if row["eye"] == "R":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "RPupil_normalized"] = interp_Rpupil
                    
                # if these were zeros this would cause the interpolation to fail, but they're not
                #print(f"Pupil dilation at t1: {LPupil[y_ind_L[0]] if row['eye'] == 'L' else RPupil[y_ind_R[0]]}")
                #print(f"Pupil dilation at t4: {LPupil[y_ind_L[1]] if row['eye'] == 'L' else RPupil[y_ind_R[1]]}")

            
    dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_num[-3:]}_r{run}_Sample_Interpolated.csv")
    return dfSamples
    
def interpolate_pupil_over_blinks(dfSamples, dfBlink, subject_num, run):
    """
    Interpolate left and right pupil sizes over blink periods. Modifies the
    dataframe of samples in place to change pupil dilation values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Changed to owrk with normalized pupil values
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
    #this_sub_run_blinks = dfBlink[(dfBlink["Subject"] == subject_num) & 
                                #(dfBlink["run_num"] == run)]
    this_sub_run_blinks = dfBlink
    # get the time of every sample
    sample_time = this_sub_run_samples['tSample'].to_numpy()
    #print(sample_time[10999:11999])
    LPupil = np.array(this_sub_run_samples['LPupil_normalized'])
    RPupil = np.array(this_sub_run_samples['RPupil_normalized'])
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
                    dfSamples.loc[this_sub_run_samples.index[mask], "LPupil_normalized"] = interp_Lpupil
                    # Print values after the update
                    #print("Values after update (LPupil):")
                    #print(dfSamples.loc[indices_to_update, "LPupil"])
                if row["eye"] == "R":
                    #print(f"Updating indices: {this_sub_run_samples.index[mask]}")
                    dfSamples.loc[this_sub_run_samples.index[mask], "RPupil_normalized"] = interp_Rpupil

                    
    dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_num[-3:]}_r{run}_Sample_Interpolated.csv")
    
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
    this_sub_run_blinks = dfBlink
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
                if valid_interpolation:
                    y_LX = LX[y_ind_LX]
                    y_LY = LY[y_ind_LY]
                    # generate the spl model using the time and pupil size
                    spl_LX = CubicSpline(x, y_LX)
                    spl_LY = CubicSpline(x, y_LY)
                    
                    mask = (sample_time > t2) & (sample_time <t3)
                    x = sample_time[mask]
                    interp_LX = spl_LX(x)
                    interp_LY = spl_LY(x)
                    # update
                    dfSamples.loc[this_sub_run_samples.index[mask], "LX"] = interp_LX
                    dfSamples.loc[this_sub_run_samples.index[mask], "LY"] = interp_LY
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
                if valid_interpolation:
                    y_RX = RX[y_ind_RX]
                    y_RY = RY[y_ind_RY]
                    # generate the spl model using the time and pupil size
                    spl_RX = CubicSpline(x, y_RX)
                    spl_RY = CubicSpline(x, y_RY)
                    
                    mask = (sample_time > t2) & (sample_time <t3)
                    x = sample_time[mask]
                    interp_RX = spl_RX(x)
                    interp_RY = spl_RY(x)
                    # update
                    dfSamples.loc[this_sub_run_samples.index[mask], "RX"] = interp_RX
                    dfSamples.loc[this_sub_run_samples.index[mask], "RY"] = interp_RY

                
    dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_num[-3:]}_r{run}_Sample_Interpolated_Pupil_Coord.csv")

def interpolate_coordinates_over_blinksv2(dfSamples, dfBlink, dfSaccades, subject_num, run):
    """
    Interpolate left and right x and y eye coordinates over blink periods. Modifies the
    dataframe of samples in place to change coordinate values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Uses saccades to find times to interpolate over. Contains adjustments recommended
    through conversation with Dr. J.
    
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
    this_sub_run_blinks = dfBlink
    # get the time of every sample
    sample_time = this_sub_run_samples['tSample'].to_numpy()
    #print(sample_time[10999:11999])
    LX = np.array(this_sub_run_samples['LX'])
    LY = np.array(this_sub_run_samples['LY'])
    RX = np.array(this_sub_run_samples['RX'])
    RY = np.array(this_sub_run_samples['RY'])
    # declare blink offset: 50ms added to the start and end of a blink
    blink_off = 0#50
    
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
                print("merging blinks")
                # update blink end time if merging blinks
                b_end = nrow["tEnd"] + blink_off
                # merge two blinks into a longer one
                update_start = False
                continue
            
        update_start = True
        # define time for interpolation
        
        # get the blink duration
        blink_dur = b_end - b_start
        update_start = True
        # define time for interpolation
        # t1 ==blink dur== t2 ==blink dur== t3 ==blink dur== t4
        t2 = b_start
        t3 = b_end
        # set t1 to be the end time of the last saccade before the blink
        #get all saccades before this blink
        previous_saccades = dfSaccades[dfSaccades["tEnd"] < b_start]
        # get last saccade before this blink
        last_saccade_tEnd = previous_saccades["tEnd"].max()
        # set t4 to be the start time of the first saccade after the blink
        # get all saccades after this blink
        later_saccades = dfSaccades[dfSaccades["tStart"] > b_end]
        # get the first saccade after this blink
        first_saccade_tStart = later_saccades["tStart"].min()
        t1 = last_saccade_tEnd
        t4 = first_saccade_tStart
        # check for missing vals in t1 or t4 and use buffer as fallback
        if pd.isna(t1):
            print("t1 is na, using blink duration")
            t1 = t2 - blink_dur
        if pd.isna(t4):
            print("t4 is na, using blink duration")
            t4 = t3 + blink_dur
        #print("t1", t1, "t2", t2, "t3", t3, "t4", t4)
        
        
        if (t1 > sample_time[0]) and (t4 < sample_time[-1]):
            # select time and pupil size for interpolation
            #x = [t1, t2, t3, t4]
            x = [t1, t4]
            # use t1 and t4 and do linear interpolation (change for l and r)
            # interpolation for LX
            if row["eye"] == "L":
                y_ind_LX = []
                y_ind_LY = []
                for t in x:
                    if t in sample_time:
                        y_ind_LX.append(np.where(sample_time==t)[0][0])
                        y_ind_LY.append(np.where(sample_time==t)[0][0])
                        #print("indLX", y_ind_LX)
                        #print("ind_LY", y_ind_LY)
                        #print("x", x)
                    else:
                        print(f"No exact match found for time {t}. Skipping this blink.")
                        valid_interpolation = False
                        break 
                    #closest_index = np.argmin(np.abs(sample_time - t))
                    #y_ind_L.append(closest_index)
                if valid_interpolation:
                    # old method
                    #y_LX = LX[y_ind_LX]
                    #y_LY = LY[y_ind_LY]
                    # get y values at t1 and t4
                    y_LX = [LX[y_ind_LX[0]], LX[y_ind_LX[1]]]
                    y_LY = [LY[y_ind_LY[0]], LY[y_ind_LY[1]]]
                    
                    # generate the spl model using the time and pupil size
                    #spl_LX = CubicSpline(x, y_LX)
                    #spl_LY = CubicSpline(x, y_LY)
                    lin_LX = interp1d(x, y_LX)
                    lin_LY = interp1d(x, y_LY)

                    
                    # using only t1 and t4 in mask
                    #mask = (sample_time > t2) & (sample_time <t3)
                    #mask = (sample_time > t1) & (sample_time <t4)
                    # base mask on closest index instead of exact
                    idx_t1 = np.argmin(np.abs(sample_time - t1))
                    idx_t4 = np.argmin(np.abs(sample_time - t4))
                    mask = (sample_time > sample_time[idx_t1]) & (sample_time < sample_time[idx_t4])
                    
                    x = sample_time[mask]
                    #print("xmask", x)
                    #interp_LX = spl_LX(x)
                    #interp_LY = spl_LY(x)
                    # linear interpolation
                    interp_LX = lin_LX(x)
                    interp_LY = lin_LY(x)
                    # update
                    dfSamples.loc[this_sub_run_samples.index[mask], "LX"] = interp_LX
                    dfSamples.loc[this_sub_run_samples.index[mask], "LY"] = interp_LY
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
                if valid_interpolation:
                    y_RX = RX[y_ind_RX]
                    y_RY = RY[y_ind_RY]
                    # generate the spl model using the time and pupil size
                    #spl_RX = CubicSpline(x, y_RX)
                    #spl_RY = CubicSpline(x, y_RY)
                    lin_RX = interp1d(x, y_RY)
                    lin_RY = interp1d(x, y_RY)
                    
                    #mask = (sample_time > t2) & (sample_time <t3)
                    # have mask for t1 and t4
                    #mask = (sample_time > t1) & (sample_time <t4)
                    # base mask on closest index instead of exact
                    idx_t1 = np.argmin(np.abs(sample_time - t1))
                    idx_t4 = np.argmin(np.abs(sample_time - t4))
                    mask = (sample_time > sample_time[idx_t1]) & (sample_time < sample_time[idx_t4])
                    x = sample_time[mask]
                    #interp_RX = spl_RX(x)
                    #interp_RY = spl_RY(x)
                    interp_RX = lin_RX(x)
                    interp_RY = lin_RY(x)
                    # update
                    dfSamples.loc[this_sub_run_samples.index[mask], "RX"] = interp_RX
                    dfSamples.loc[this_sub_run_samples.index[mask], "RY"] = interp_RY

                
    #dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_num[-3:]}_r{run}_Sample_Interpolated_Pupil_Coord.csv")
    return dfSamples

def interpolate_coordinates_over_blinksv2_nomerge(dfSamples, dfBlink, dfSaccades, subject_num, run):
    """
    Interpolate left and right x and y eye coordinates over blink periods. Modifies the
    dataframe of samples in place to change coordinate values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Uses saccades to find times to interpolate over. Contains adjustments recommended
    through conversation with Dr. J.
    
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
    this_sub_run_blinks = dfBlink
    # get the time of every sample
    sample_time = this_sub_run_samples['tSample'].to_numpy()
    #print(sample_time[10999:11999])
    LX = np.array(this_sub_run_samples['LX'])
    LY = np.array(this_sub_run_samples['LY'])
    RX = np.array(this_sub_run_samples['RX'])
    RY = np.array(this_sub_run_samples['RY'])

    
    # iterate throu each row of blink dataframe
    print("num blinks for this subject and run: ", len(this_sub_run_blinks))
    print("num samples for this subject and run: ", len(this_sub_run_samples))
    for index in np.arange(len(this_sub_run_blinks)):
        # reset flag
        valid_interpolation = True
        row = this_sub_run_blinks.iloc[index]
        # get the start and end time
        b_start = row['tStart'] 
        b_end = row['tEnd']
            

        
        # get the blink duration
        blink_dur = b_end - b_start


        # set t1 to be the end time of the last saccade before the blink
        #get all saccades before this blink
        previous_saccades = dfSaccades[dfSaccades["tEnd"] < b_start]
        # get last saccade before this blink
        last_saccade_tEnd = previous_saccades["tEnd"].max()
        # set t4 to be the start time of the first saccade after the blink
        # get all saccades after this blink
        later_saccades = dfSaccades[dfSaccades["tStart"] > b_end]
        # get the first saccade after this blink
        first_saccade_tStart = later_saccades["tStart"].min()
        t1 = last_saccade_tEnd
        t4 = first_saccade_tStart
        # check for missing vals in t1 or t4 and use buffer as fallback
        if pd.isna(t1):
            print("t1 is na, using blink duration")
            t1 = b_start
        if pd.isna(t4):
            print("t4 is na, using blink duration")
            t4 = b_end
        #print("t1", t1, "t2", t2, "t3", t3, "t4", t4)
        
        
        if (t1 > sample_time[0]) and (t4 < sample_time[-1]):
            # select time and pupil size for interpolation
            #x = [t1, t2, t3, t4]
            x = [t1, t4]
            # use t1 and t4 and do linear interpolation (change for l and r)
            # interpolation for LX
            if row["eye"] == "L":
                y_ind_LX = []
                y_ind_LY = []
                for t in x:
                    if t in sample_time:
                        y_ind_LX.append(np.where(sample_time==t)[0][0])
                        y_ind_LY.append(np.where(sample_time==t)[0][0])
                        #print("indLX", y_ind_LX)
                        #print("ind_LY", y_ind_LY)
                        #print("x", x)
                    else:
                        print(f"No exact match found for time {t}. Skipping this blink.")
                        valid_interpolation = False
                        break 
                    #closest_index = np.argmin(np.abs(sample_time - t))
                    #y_ind_L.append(closest_index)
                if valid_interpolation:
                    # old method
                    #y_LX = LX[y_ind_LX]
                    #y_LY = LY[y_ind_LY]
                    # get y values at t1 and t4
                    y_LX = [LX[y_ind_LX[0]], LX[y_ind_LX[1]]]
                    y_LY = [LY[y_ind_LY[0]], LY[y_ind_LY[1]]]
                    
                    # generate the spl model using the time and pupil size
                    #spl_LX = CubicSpline(x, y_LX)
                    #spl_LY = CubicSpline(x, y_LY)
                    lin_LX = interp1d(x, y_LX)
                    lin_LY = interp1d(x, y_LY)

                    
                    # using only t1 and t4 in mask
                    #mask = (sample_time > t2) & (sample_time <t3)
                    mask = (sample_time > t1) & (sample_time <t4)
                    
                    x = sample_time[mask]
                    #print("xmask", x)
                    #interp_LX = spl_LX(x)
                    #interp_LY = spl_LY(x)
                    # linear interpolation
                    interp_LX = lin_LX(x)
                    interp_LY = lin_LY(x)
                    # update
                    dfSamples.loc[this_sub_run_samples.index[mask], "LX"] = interp_LX
                    dfSamples.loc[this_sub_run_samples.index[mask], "LY"] = interp_LY
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
                if valid_interpolation:
                    y_RX = RX[y_ind_RX]
                    y_RY = RY[y_ind_RY]
                    # generate the spl model using the time and pupil size
                    #spl_RX = CubicSpline(x, y_RX)
                    #spl_RY = CubicSpline(x, y_RY)
                    lin_RX = interp1d(x, y_RY)
                    lin_RY = interp1d(x, y_RY)
                    
                    #mask = (sample_time > t2) & (sample_time <t3)
                    # have mask for t1 and t4
                    mask = (sample_time > t1) & (sample_time <t4)
                    x = sample_time[mask]
                    #interp_RX = spl_RX(x)
                    #interp_RY = spl_RY(x)
                    interp_RX = lin_RX(x)
                    interp_RY = lin_RY(x)
                    # update
                    dfSamples.loc[this_sub_run_samples.index[mask], "RX"] = interp_RX
                    dfSamples.loc[this_sub_run_samples.index[mask], "RY"] = interp_RY

                
    #dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_num[-3:]}_r{run}_Sample_Interpolated_Pupil_Coord.csv")
    return dfSamples

def preprocess(dfSamples):
    """
    Performs preprocessing on the dataframe of raw samples for each subject. 
    Drops rows with missing run num and creates the tSample_normalized column,
    which is a column of normalized time such that each run begins at time 0.
    
    Removed normalize pupil bc doing this before interpolation now

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
    
    # normalize pupils - done before interpolation, don't do again
    #dfSamples = normalize_pupil(dfSamples)
    
    # make subject values strings
    dfSamples["Subject"] = dfSamples["Subject"].astype(str)
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
    data = dfSamples[["tSample_normalized", "RX", "RY", "RPupil_normalized",
                      "LX", "LY", "LPupil_normalized", "is_MW"]]#"Lblink", "Rblink", "is_MW"]]
    
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
    Generates a line plot of the normalized pupil dilation over time, averaged over subjects
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
    
    LPupil_over_time = dfSamples.groupby("tSample_normalized")["LPupil_normalized"].mean()
    
    RPupil_over_time = dfSamples.groupby("tSample_normalized")["RPupil_normalized"].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(LPupil_over_time.index, LPupil_over_time.values, label="Left Pupil")
    plt.plot(RPupil_over_time.index, RPupil_over_time.values, alpha=.4, label="Right Pupil")
    plt.xlabel("Normalized Time (ms from run start)")
    plt.ylabel("Within-Subject Normalized Dilation of Pupil (averaged over subjects and runs)")
    plt.title("Pupil Dilation Over Time")
    plt.legend()
    
    plt.savefig("plots/dilation_over_time_avg_sub_run.png")



def pupil_subject_run(dfSamples, subjects):
    """
    Generates one plot for each subject with subplots for each run. Each subplot
    shows the normalized pupil dilation over time for that subject and run. Each plot is 
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
                               this_sub_run_df["LPupil_normalized"], label="Left Pupil")
            axes[run_num-1].plot(this_sub_run_df["tSample_normalized"],
                               this_sub_run_df["RPupil_normalized"], alpha = .4, label="Right Pupil")
            # set labels
            axes[run_num-1].set_title(f"Run {run_num}")
            axes[run_num-1].set_xlabel("Normalized Time (ms from run start)")
            axes[run_num-1].set_ylabel("Within-Subject Normalized Pupil Dilation")
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

def normalize_pupil_pre_interp(dfSamples):
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
    dfSamples["LPupil_normalized"] = dfSamples["LPupil"].transform(lambda x: x / x.mean())
    
    # Normalize RPupil for each subject
    dfSamples["RPupil_normalized"] = dfSamples["RPupil"].transform(lambda x: x / x.mean())
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
        
    
def coord_over_time(dfSamples):
    """
    Generates line plots of the x and y eye coordinates respectively over time, averaged over subjects
    and runs. The line plots are then saved to the plots folder within the current
    working directory.

    Parameters
    ----------
    dfSamples : Dataframe
        The dataframe of samples for each subject.

    Returns
    -------
    None
    """
    
    LX_over_time = dfSamples.groupby("tSample_normalized")["LX"].mean()
    LY_over_time = dfSamples.groupby("tSample_normalized")["LY"].mean()
    
    RX_over_time = dfSamples.groupby("tSample_normalized")["RX"].mean()
    RY_over_time = dfSamples.groupby("tSample_normalized")["RY"].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(LX_over_time.index, LX_over_time.values,  alpha =.4, label="Left Eye X Coordinates", color="red")
    plt.plot(LY_over_time.index, LY_over_time.values,  alpha=.4, label="Left Eye Y Coordinates", color="red")
    plt.plot(RX_over_time.index, RX_over_time.values, label="Right Eye X Coordinates", color="green")
    plt.plot(RY_over_time.index, RY_over_time.values, label="Right Eye Y Coordinates", color="green")
    plt.xlabel("Normalized Time (ms from run start)")
    plt.ylabel("Coordinates")
    plt.title("Eye Gaze Coordinates Over Time (averaged over subjects and runs)")
    plt.legend()
    
    plt.savefig("plots/coord_over_time_avg_sub_run.png")
#%% Gaze coordinates over time (averaged over subjects and runs)
# just to check for artifacts


#%% Interpolate Pupil
# may need to be updated to utilize saccades instead of a blink buffer
# run interpolation on samples csv for each subject and run
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
data_path = "E:\\" 
dfBlink = pd.read_csv(f"{data_path}MW_Classifier_Data\\all_s_blinks.csv")
for subject in subjects:
    for run in range(1,6):
        subject_str = str(subject)
        dfSamples = pd.read_csv(f"{data_path}/CSV_Samples/{subject}/s{subject_str[-3:]}_r{run}_Sample.csv")
        interpolate_pupil_over_blinks(dfSamples, dfBlink, subject, run)
        

#%%
# investigate issue with non-finite values in eye coordinates
s10014_inter_r1 = pd.read_csv("E:\\CSV_Samples\\10014\\s014_r1_Sample_Interpolated_Pupil.csv")
print("missing values")
print(s10014_inter_r1[['LX', 'LY', 'RX', 'RY']].isna().sum())
print("inf values")
print(s10014_inter_r1[['LX', 'LY', 'RX', 'RY']].apply(lambda x: np.isinf(x).sum()))


#%% Interpolate Coordinates, updating the csvs that already went through pupil interpolation
# hold off on this until I can discuss infinite values with lab - how to proceed with them? 
# set to na then interpolate over? linear interpolation?

# before doing this, make a plot of the coordinates over time on the not interpolated
# data and save in pre artifact removal folder to show a before and after
for subject in subjects:
    for run in range(1,6):
        subject_str = str(subject)
        dfSamples = pd.read_csv(f"E:\\CSV_Samples\\{subject}\\s{subject_str[-3:]}_r{run}_Sample_Interpolated.csv")
        interpolate_coordinates_over_blinks(dfSamples, dfBlink, subject, run)

#%% extract
# done, these are csvs on lexar (interpolated csvs are interpolation with errors)
# currently set to extract from interpolated_pupil files
# create raw samples from those interpolated csvs (extract_raw_samples)
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]

for subject in subjects:
    folder_path = f"E:\\CSV_Samples\\{subject}"
    extract(subject, folder_path)

#%% Preprocess
# redo process starting here after changing drive format
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
data_path = "E:\\MW_Classifier_Data\\"
dfSamples = load_raw_samples(data_path, subjects)
dfSamples = preprocess(dfSamples)

#%%
# Drop "Unnamed: 0", "LPupil", "RPupil"
dfSamples = dfSamples.drop(["Unnamed: 0", "LPupil", "RPupil"], axis=1)
print(dfSamples.columns)
#%% save processed file to new csv
chunksize = 100000
dfSamples.to_csv("E:\\MW_Classifier_Data\\all_subjects_interpolated_pupil.csv", chunksize=chunksize, index=False)

#%% 
# get subject and run num dtypes in interpolated samples
print(dfSamples["Subject"])
print(dfSamples["run_num"])

#both are numeric

#%%

dfSamples = pd.read_csv("E:\\MW_Classifier_Data\\all_subjects_interpolated_pupil.csv")
#%% Plot
correlation_heatmap(dfSamples)
mw_freq_by_subject(dfSamples)
mw_over_time(dfSamples)
pupil_over_time(dfSamples)
pupil_subject_run(dfSamples, subjects)
eye_coordinates(dfSamples, subjects)


#%% Get subject datatype
data_path = "/Volumes/Lexar/MW_Classifier_Data/" 
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]

dfSamples = pd.read_csv(f"{data_path}all_subjects_data_no_interpolation.csv")
print(dfSamples["Subject"])



#%%
data_path = "/Volumes/Lexar/"
# check to see if there are any values other than L and R for eye in blinks
dfBlink = pd.read_csv(f"{data_path}MW_Classifier_Data/all_s_blinks.csv")
print(dfBlink["eye"].unique())

# subject is int in dfSamples when not interpolated
left_blinks = dfBlink[dfBlink["eye"] == "L"]
right_blinks = dfBlink[dfBlink["eye"] == "R"]

# get blink durations
left_blink_durations = left_blinks["tEnd"] - left_blinks["tStart"]
right_blink_durations = right_blinks["tEnd"] - right_blinks["tStart"]

# plot distribution
plt.figure(figsize=(10, 6))
plt.hist(left_blink_durations, bins=50, alpha=0.5, label='Left Eye Blink Durations')
plt.hist(right_blink_durations, bins=50, alpha=0.5, label='Right Eye Blink Durations')
plt.xlabel('Blink Duration (ms)')
plt.ylabel('Frequency')
plt.title('Distribution of Blink Durations for Left and Right Eyes')
plt.legend()
plt.show()

# we do not have data for many R eye blinks in the 1300 ms-2000 ms range
# but we do have data for L eye blinks of that length. just use 
# L eye for analysis



#%%

dfSamples_no_interpolation = pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_subjects_data_no_interpolation.csv")

#%%
print(dfSamples_no_interpolation.columns)

# how many ms in each run
# Group by both Subject and run number and count the number of rows for each run
rows_per_subject_run = dfSamples_no_interpolation.groupby(['Subject', 'run_num']).size()

# Calculate the average number of rows per run across all subjects
average_rows_per_run = rows_per_subject_run.mean()

print(average_rows_per_run)
#%%
dfSamples_no_interpolation = normalize_pupil(dfSamples_no_interpolation)

dfSamples_no_interpolation.columns

#%%

dfSamples_no_interpolation["Subject"].unique()

#%% 

dfSamples_no_interpolation = dfSamples_no_interpolation.drop(["Unnamed: 0", "LPupil", "RPupil"], axis=1)
print(dfSamples_no_interpolation["Subject"])
#%%
dfSamples_no_interpolation.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_subjects_data_no_interpolation.csv", chunksize=100000, index=False)
#%% Plot
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
dfSamples_no_interpolation["Subject"] = dfSamples_no_interpolation['Subject'].astype(str)
#correlation_heatmap(dfSamples_no_interpolation)
mw_freq_by_subject(dfSamples_no_interpolation)
mw_over_time(dfSamples_no_interpolation)
#pupil_over_time(dfSamples_no_interpolation)
#pupil_subject_run(dfSamples_no_interpolation, subjects)
#eye_coordinates(dfSamples_no_interpolation, subjects)

#%%
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
data_path = "/Volumes/brainlab/Mindless Reading/DataCollection/"
dfBlinks = load_blink_data(subjects, data_path)

#%% load all blinks
dfBlink = pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_blinks_9_30.csv")
#%%  get info about data 
# set min as the shortest blink duration
# fewer, wider bins for the upper range to avoid skewing plot because of outliers
print(dfBlink["Subject"])
print(dfBlink.columns)

# drop subject 10127 bc only one eye data
dfBlink = dfBlink.loc[dfBlink["Subject"] != 10127]

left_blinks = dfBlink[dfBlink["eye"] == "L"]
right_blinks = dfBlink[dfBlink["eye"] == "R"]

# get blink durations
left_blink_durations = left_blinks["duration"]
right_blink_durations = right_blinks["duration"]

# get info about blink durs
print(left_blink_durations.describe())
print(right_blink_durations.describe())

"""
L blinks
mean        213.303411
std        1373.980800
min           1.000000
25%          77.000000
50%         104.000000
75%         139.000000
max      105310.000000

R blinks
mean       184.276158
std        604.491120
min          1.000000
25%         75.000000
50%        100.000000
75%        136.000000
max      34827.000000

Most blinks are between 74-140ms (IQR) - have fine bins here and wider ones for outliers
"""
print(dfBlink["Subject"].unique())
#%% make bin edges array
# higher third # = smaller bins
# little bins for small blinks 
# medium bins for medium blinks
# large bins for our large, outlier blinks
#bin_edges = [np.linspace(0,140,40), np.linspace(140,105310, 30)]
#bin_edges = np.concatenate(bin_edges)

# make 50 log spaced bin edges from min to max
bin_edges = np.logspace(np.log10(1), np.log10(105310), 50)
#%% plot blink dist
# plot distribution
# provide array of bin edges - we probably have one long outlier for l eye blink
plt.figure(figsize=(10, 6))
plt.hist(left_blink_durations, bins=bin_edges, alpha=0.5, label='Left Eye Blink Durations')
plt.hist(right_blink_durations, bins=bin_edges, alpha=0.5, label='Right Eye Blink Durations')
plt.xscale("log")
plt.xlabel('Blink Duration (ms)')
plt.ylabel('Frequency')
plt.title('Distribution of Blink Durations for Left and Right Eyes')
plt.legend()
plt.show()

# from excel countif on all blinks file- 16799 Lblinks, 18790 Rblinks
# so, more right eye blinks, but shorter than left eye blinks. Why are they different?
#%% CURRENT STARTS HERE- dated
# re run interpolation- try only using files in gbl
# uses v2- interpolates over saccades
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125,10127]
for subject_num in subjects:
    for run_num in range(1,6):
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_sample_filenames = [file for file in filenames if (file.endswith("Sample.csv") and not file.startswith("._"))]
        sample_file = matching_sample_filenames[0]
        print("sample file", sample_file)
        matching_blink_filenames = [file for file in filenames if (file.endswith("Blink.csv") and not file.startswith("._"))]
        blink_file = matching_blink_filenames[0]
        print("blink file", blink_file)
        matching_saccade_filenames = [file for file in filenames if (file.endswith("Saccade.csv") and not file.startswith("._"))]
        saccade_file = matching_saccade_filenames[0]
        print("saccade file", saccade_file)
        dfSamples = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{sample_file}")
        dfBlink = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{blink_file}")
        dfSaccades = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{saccade_file}")
        interpolate_pupil_over_blinksv2(dfSamples, dfBlink, dfSaccades, subject_str, run_num)
        
#%%
# missing values for eye coordinates are dispersed over run durations
# didnt finish generating plots bc i could see the trend.. that they are dispersed over runs
# see how long the sequences of missing values are and where in the runs they are
# load in samples for this subject and run
# plot missing values as a function of tsample 

subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
for subject_num in subjects:
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    fig.suptitle(f"Missing Eye Coordinates for Subject {subject_num}")
    axes = axes.flatten()
    for run_num in range(1,6): # 5 runs for each subject
        # get subset of dataframe for this subject and run
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_sample_filenames = [file for file in filenames if (file.endswith("Sample.csv") and not file.startswith("._"))]
        sample_file = matching_sample_filenames[0]
        print("sample file", sample_file)
        # get subset of dataframe for this subject and run
        this_sub_run_df = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{sample_file}")
        # get missing L eye data (either lx or ly is missing, but if one is missing the other is prob missing)
        missing_l_data = this_sub_run_df[["LX", "LY"]].isna().any(axis=1).astype(int)
        missing_r_data = this_sub_run_df[["RX", "RY"]].isna().any(axis=1).astype(int)
        
        missing_l_data = missing_l_data == 1
        missing_r_data = missing_r_data == 1
         # plot pupil dilation for this subject and run
        axes[run_num-1].scatter(this_sub_run_df["tSample"][missing_l_data],
                            [1] * sum(missing_l_data), label="Left Eye", color="red", alpha=.4)
        axes[run_num-1].scatter(this_sub_run_df["tSample"][missing_r_data],
                            [1] * sum(missing_r_data), label="Right Eye", color="green", alpha=.4)
         # set labels
        axes[run_num-1].set_xlim([this_sub_run_df["tSample"].min(), this_sub_run_df["tSample"].max()])
        axes[run_num-1].set_title(f"Run {run_num}")
        axes[run_num-1].set_xlabel("tSample")
        axes[run_num-1].set_ylabel("Missing Coordinates")
     
     # get rid of the empty subplot
    fig.delaxes(axes[5])
     
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig(f"plots/Subject{subject_num}_missing_coordinates.png")
    plt.close(fig)
    
#%%
# get max consecutive missing values to see if interpolation over na could be appropriate
# if less than 250 max then we can do linear interpolation because we don't see many saccades in that timeframe
def max_consecutive_missing(is_missing):
    # Find consecutive missing values by grouping by shifts
    consecutive_groups = (is_missing != is_missing.shift()).cumsum()
    
    # Count the size of each missing group and get the max
    return is_missing.groupby(consecutive_groups).sum().max()

subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
max_consec_missing = []
for subject_num in subjects:
    for run_num in range(1,6): # 5 runs for each subject
        # get subset of dataframe for this subject and run
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_sample_filenames = [file for file in filenames if (file.endswith("Sample.csv") and not file.startswith("._"))]
        sample_file = matching_sample_filenames[0]
        print("sample file", sample_file)
        # get subset of dataframe for this subject and run
        this_sub_run_df = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{sample_file}")
        # get missing L eye data (either lx or ly is missing, but if one is missing the other is prob missing)
        missing_l_data = this_sub_run_df[["LX", "LY"]].isna().any(axis=1)
        missing_r_data = this_sub_run_df[["RX", "RY"]].isna().any(axis=1)
        # Find the maximum number of consecutive missing values for L and R eye
        max_missing_L = max_consecutive_missing(missing_l_data)
        max_missing_R = max_consecutive_missing(missing_r_data)
        
        # Store the result in the list
        max_consec_missing.append({
            'Subject': subject_num,
            'Run': run_num,
            'Max_Consecutive_Missing_L': max_missing_L,
            'Max_Consecutive_Missing_R': max_missing_R
        })

max_missing_df = pd.DataFrame(max_consec_missing)

print(max_missing_df)
        
# if < 250 max, do linear interp over missing values (pandas method)
# many are >250 :(
# then do regular coordinate interpolation after updating function to use fallback
# then create the df of interpolated samples and plot to check work

#%%
# check if missing coordinates coincide with blink periods
# if they do, forward fill the coordinates because we can assume eye
# coordinates didnt change during blinks

# ALL MISSING COORDINATES ARE WITHIN BLINK PERIODS
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]

all_missing_R_outside_blink = []
all_missing_L_outside_blink = []

for subject_num in subjects:
    for run_num in range(1,6): # 5 runs for each subject
        # get subset of dataframe for this subject and run
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_sample_filenames = [file for file in filenames if (file.endswith("Sample.csv") and not file.startswith("._"))]
        sample_file = matching_sample_filenames[0]
        print("sample file", sample_file)
        matching_blink_filenames = [file for file in filenames if (file.endswith("Blink.csv") and not file.startswith("._"))]
        blink_file = matching_blink_filenames[0]
        print("blink file", blink_file)
        this_sub_run_blinks = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{blink_file}")
        # get subset of dataframe for this subject and run
        this_sub_run_df = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{sample_file}")
        # get missing L eye data (either lx or ly is missing, but if one is missing the other is prob missing)
        this_sub_run_df["missing_l"] = this_sub_run_df[["LX", "LY"]].isna().any(axis=1)
        this_sub_run_df["missing_r"] = this_sub_run_df[["RX", "RY"]].isna().any(axis=1)
        # set missing during blink cols to false
        this_sub_run_df["missing_during_blink_L"] = False
        this_sub_run_df["missing_during_blink_R"] = False
        
        for index, blink in this_sub_run_blinks.iterrows():
            # Flag missing gaze coordinates during the blink period for the left eye
            this_sub_run_df.loc[(this_sub_run_df['tSample'] >= blink['tStart']) &
                          (this_sub_run_df['tSample'] <= blink['tEnd']) &
                          this_sub_run_df['missing_l'], 'missing_during_blink_L'] = True
            
            # Flag missing gaze coordinates during the blink period for the right eye
            this_sub_run_df.loc[(this_sub_run_df['tSample'] >= blink['tStart']) &
                          (this_sub_run_df['tSample'] <= blink['tEnd']) &
                          this_sub_run_df['missing_r'], 'missing_during_blink_R'] = True
        # save the number of missing coords that arent in blink periods
        
        count_missingR_not_during_blink = this_sub_run_df[(this_sub_run_df["missing_r"] == True) &
                                                          (this_sub_run_df["missing_during_blink_R"]==False)].shape[0]
        count_missingL_not_during_blink = this_sub_run_df[(this_sub_run_df["missing_l"] == True) &
                                                          (this_sub_run_df["missing_during_blink_L"]==False)].shape[0]
        all_missing_R_outside_blink.append(count_missingR_not_during_blink)
        all_missing_L_outside_blink.append(count_missingL_not_during_blink)

print("Number of missing L eye coordinates outside of blink periods")
print(sum(all_missing_L_outside_blink))

print("Number of missing R eye coordinates outside of blink periods")
print(sum(all_missing_R_outside_blink))
#%%
# forward fill the missing coordinates and run coordinate interpolation for each subject and run
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]

for subject_num in subjects:
    for run_num in range(1,6):
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_blink_filenames = [file for file in filenames if (file.endswith("Blink.csv") and not file.startswith("._"))]
        blink_file = matching_blink_filenames[0]
        print("blink file", blink_file)
        matching_saccade_filenames = [file for file in filenames if (file.endswith("Saccade.csv") and not file.startswith("._"))]
        saccade_file = matching_saccade_filenames[0]
        print("saccade file", saccade_file)
        # feed in file that already had pupil interpolation
        dfSamples = pd.read_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_str[-3:]}_r{run_num}_Sample_Interpolated.csv")
        dfBlink = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{blink_file}")
        dfSaccade = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{saccade_file}")

        # forward fill missing coordinates
        dfSamples[["LX", "LY", "RX", "RY"]] = dfSamples[["LX", "LY", "RX", "RY"]].fillna(method="ffill")

        # interpolate
        interpolate_coordinates_over_blinksv2(dfSamples, dfBlink, dfSaccades, subject_str, run_num)

#%%
# extract interpolated_pupil_coord
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]

for subject in subjects:
    folder_path = f"/Volumes/Lexar/CSV_Samples/{subject}"
    extract(subject, folder_path)
    
#%% 
# preprocess
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
data_path = "/Volumes/Lexar/MW_Classifier_Data/"
dfSamples = load_raw_samples(data_path, subjects)
dfSamples = preprocess(dfSamples)
print(dfSamples.columns)

#%%
# drop cols if needed (unnamed, lpupil, rpupil)
dfSamples = dfSamples.drop(["Unnamed: 0.1", "Unnamed: 0", "LPupil", "RPupil"],axis=1)

#%% 
# save to csv
dfSamples.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_subjects_interpolated_pupil_coord.csv")
#%% 
print(dfSamples["Subject"])

#%% 
# read csv
chunks = []
chunk_count = 0
total_rows = 0
chunksize = 500000
for chunk in pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_subjects_interpolated_pupil_coord.csv", chunksize=chunksize):
    chunks.append(chunk)
    chunk_count += 1
    total_rows += len(chunk)
    # print progress
    print(f"Processed chunk {chunk_count}, total rows processed: {total_rows}")
# concat chunks into df
dfSamples = pd.concat(chunks,ignore_index=True)
#%% Current ends here

dfSamples["Subject"] = dfSamples["Subject"].astype(str)
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
print("creating corr heatmap")
correlation_heatmap(dfSamples)
print("freq by subject")
mw_freq_by_subject(dfSamples)
print("mw over time")
mw_over_time(dfSamples)
print("pupil over time")
pupil_over_time(dfSamples)
print("pupil subject run")
pupil_subject_run(dfSamples, subjects)
print("eye coords")
eye_coordinates(dfSamples, subjects)
print("coord over time")
coord_over_time(dfSamples)


#%%
# check for missing values in samples
for subject_num in subjects:
    for run_num in range(1,6):
        subject_str = str(subject_num)
        dfSamples = pd.read_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_str[-3:]}_r{run_num}_Sample_Interpolated.csv")
        LX_na = dfSamples["LX"].isna().sum()
        LY_na = dfSamples["LY"].isna().sum()
        RX_na = dfSamples["RX"].isna().sum()
        RY_na = dfSamples["RY"].isna().sum()
        print(f"Subject {subject_num}, run {run_num}")
        print("total rows: ",len(dfSamples))
        print(f"LX Na: {LX_na}")
        print(f"LY Na: {LY_na}")
        print(f"RX Na: {RX_na}")
        print(f"RY Na: {RY_na}")
        
# s10127 is missing all left eye data
        
#%%
# drop s10127 - leave them out of subjects list when processing
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
for subject_num in subjects:
    for run_num in range(1,6):
        subject_str = str(subject_num)
        print(subject_num, run_num)
        # drop rows with missing coordinate values from sample_interpolated files 
        interpolated_samples = pd.read_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_str[-3:]}_r{run_num}_Sample_Interpolated.csv")
        interpolated_samples = interpolated_samples.dropna(subset=["LX", "LY", "RX", "RY"])
        # check df length
        print(len(interpolated_samples))
        # resave csvs 
        interpolated_samples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_str[-3:]}_r{run_num}_Sample_Interpolated.csv")


#%% coordinate interpolation - run on interpolated samples 

subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
# because we dropped some rows, some blinks will certainly be skipped

for subject_num in subjects:
    for run_num in range(1,6):
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_blink_filenames = [file for file in filenames if (file.endswith("Blink.csv") and not file.startswith("._"))]
        blink_file = matching_blink_filenames[0]
        print("blink file", blink_file)
        # feed in file that already had pupil interpolation
        dfSamples = pd.read_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_str[-3:]}_r{run_num}_Sample_Interpolated.csv")
        dfBlink = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{blink_file}")
        interpolate_coordinates_over_blinks(dfSamples, dfBlink, subject_str, run_num)
#%% extract
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]

for subject in subjects:
    folder_path = f"/Volumes/Lexar/CSV_Samples/{subject}"
    #folder_path2 = f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject}"
    extract(subject, folder_path)
    
#%%
# check a df resulting from extraction to see how to load it best
df = pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/10014_raw_interpolated_pupil_coord.csv")
print(df.dtypes)
print(df.isna().sum())
df.head()

#%% Preprocess and plot
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
data_path = "/Volumes/Lexar/MW_Classifier_Data/"
dfSamples = load_raw_samples(data_path, subjects)
dfSamples = preprocess(dfSamples) # includes pupil normalization

print(dfSamples.columns)
#%%
# Drop "Unnamed: 0", "LPupil", "RPupil"
dfSamples = dfSamples.drop(["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "LPupil", "RPupil"], axis=1)
print(dfSamples.columns)
print(dfSamples["Subject"])
print(len(dfSamples))

#%%
# save csv
dfSamples["Subject"] = dfSamples["Subject"].astype(str)
dfSamples.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_subjects_interpolated_pupil_coord.csv")
#%%
# generate plots
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
correlation_heatmap(dfSamples)
mw_freq_by_subject(dfSamples)
mw_over_time(dfSamples)
pupil_over_time(dfSamples)
pupil_subject_run(dfSamples, subjects)
eye_coordinates(dfSamples, subjects)
coord_over_time(dfSamples)

#%% 
eye_coordinates(dfSamples, subjects)
coord_over_time(dfSamples)

#%%
print(dfSamples["LPupil_normalized"].min())
print(dfSamples["RPupil_normalized"].min())

#%% save processed file to new csv if it did
chunksize = 100000
dfSamples.to_csv("E:\\MW_Classifier_Data\\all_subjects_interpolated_pupil.csv", chunksize=chunksize, index=False)

#%% load saccades and blinks

subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
data_path = "/Volumes/brainlab/Mindless Reading/DataCollection/"
all_saccades = load_saccade_data(subjects, data_path)
all_blinks = load_blink_data(subjects, data_path)
#all_saccades = pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_saccades.csv")
#all_blinks = pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_blinks_9_30.csv")

#%%
# check dtype of subjects in all saccades and all blinks - both are int
print(all_saccades["Subject"])
print(all_blinks["Subject"])

print(all_blinks.isna().sum())
print(all_blinks["tStart"])

print(len(all_blinks))

print(all_blinks["run_num"])
print(all_saccades["run_num"])

# change run num to int in each (they are object now)
all_blinks["run_num"] = all_blinks["run_num"].astype(int)
all_saccades["run_num"] = all_saccades["run_num"].astype(int)

#%% load uninterpolated data
chunks = []
chunk_count = 0
total_rows = 0
for chunk in pd.read_csv("/Volumes/brainlab/Mindless Reading/neuralnet_classifier/all_subjects_data_no_interpolation.csv", chunksize=500000):
    chunks.append(chunk)
    chunk_count += 1
    total_rows += len(chunk)
    # print progress
    print(f"Processed chunk {chunk_count}, total rows processed: {total_rows}")
# concat chunks into df
dfSamples = pd.concat(chunks,ignore_index=True)
#%% check saccades
# check the timing logged for saccade and blink to investigate if there
# is a better way to do the interpolation
# if we always find that a blink is in between two quick saccades, we can use those saccades
# as t1 and t4 in interpolation 
#all_saccades = pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_saccades.csv")
#all_blinks = pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_s_blinks_9_30.csv")
#sorted_blinks = all_blinks.sort_values(by=["Subject","run_num","tStart"])
#sorted_saccades = all_saccades.sort_values(by=["Subject","run_num","tStart"])

# adjust- instead, look for a saccade whose start time is before blink start time and end time is after blinks end time
# such that sacade start and end around blink start and end? Did I understand dr. J correctly about this bc it 
# doesnt really make sense
# its ok if there are a couple edge cases if they are on the end of trials
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]

def is_blink_between_saccades_v2(blinks_subset, saccades_subset):
    # keep track of indices for false vals too
    false_indices = []
    # initialize list to store surrounding saccades for this sub run
    # one boolean value for each blink for this sub run T if blink surrounded, F otherwise
    sub_run_sur_saccs = []
    # initialize our definition of "just after"/ "just before" (ms)
    buffer = 1000
    # iterate through rows of blinks (individual blinks)
    for index, row in blinks_subset.iterrows():
        # check if there is a saccade that starts just before this blink starts
        # and that ends just after this blink ends (within 50ms)
        # pre sacc ends before blink starts within buffer time
        this_row_pre_sacc =  np.any((saccades_subset['tEnd']<row['tStart']) &
                                    ((row["tStart"] - saccades_subset["tEnd"])<=buffer))
        # post sacc starts after blink ends (within 50 ms)
        this_row_post_sacc = np.any((saccades_subset['tStart'] > row['tEnd'])&
                                    ((saccades_subset["tStart"]- row["tEnd"]) <=buffer))
        # if this row has pre and post saccs, add true to list
        if this_row_pre_sacc and this_row_post_sacc:
            sub_run_sur_saccs.append(True)
            # else add false to list and add index to false indices list
        else:
            sub_run_sur_saccs.append(False)
            false_indices.append(index)
    return sub_run_sur_saccs, false_indices
"""
all_sub_run_sur_saccs = []
all_false_indices = []
for subject in subjects:
    for run in range(1,6):
        saccades_subset = all_saccades[(all_saccades["Subject"] == subject) & all_saccades["run_num"] == run]
        blinks_subset = all_blinks[(all_blinks["Subject"] == subject) & all_blinks["run_num"] == run]
        this_sub_run_surrounding_saccades, false_indices = is_blink_between_saccades_v2(blinks_subset, saccades_subset, subject, run)
        all_sub_run_sur_saccs.append(this_sub_run_surrounding_saccades)
        all_false_indices.append(false_indices)
"""
        
# if all false indices isnt empty, check to see if those are at the end of runs
# indices are indices in blinks file
# create a plot for each subject with subplots for each run
# showing points for each false index with run time as x axis
# x axis should range from min tStart for sub and run to max tStart for sub and run
# points should be placed for tStart for each false index

for subject_num in subjects:
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    fig.suptitle(f"Blinks not Between Saccades for Subject {subject_num}")
    axes = axes.flatten()
    for run_num in range(1,6): # 5 runs for each subject
     # get subset of dataframes for this subject and run
         this_sub_run_blinks = all_blinks[(all_blinks["Subject"] == subject_num) &
                                    (all_blinks["run_num"] == run_num)]
         # reset index in this_sub_run_blinks so we get sequential indexes and can use .loc later
         this_sub_run_blinks = this_sub_run_blinks.reset_index()
         this_sub_run_saccades = all_saccades[(all_saccades["Subject"] == subject_num) &
                                    (all_saccades["run_num"] == run_num)]
         # get time start and end
         print(len(this_sub_run_blinks))
         print(this_sub_run_blinks.isna().sum())
         time_start = this_sub_run_blinks["tStart"].min()
         time_end = this_sub_run_blinks["tStart"].max()
         print(time_start)
         print(time_end)
         # get false indices for this subject and run
         this_sub_run_surrounding_saccades, false_indices = is_blink_between_saccades_v2(this_sub_run_blinks,
                                                                                         this_sub_run_saccades)
         if len(false_indices) > 0: # only plot if there are non surrounded blinks to plot
             non_surrounded_blinks = this_sub_run_blinks.iloc[false_indices]
             # plot false indices against time
             axes[run_num-1].scatter(non_surrounded_blinks["tStart"], [1]*len(non_surrounded_blinks))
             axes[run_num-1].set_xlim(time_start, time_end)
             # set labels
             axes[run_num-1].set_title(f"Run {run_num}")
             axes[run_num-1].set_xlabel("Time (ms from first blink)")
             axes[run_num-1].set_ylabel("False index")
         else:
            print(f"There are no blinks that aren't between saccades for subject {subject_num}, run {run_num}")
     
     # get rid of the empty subplot
    fig.delaxes(axes[5])
    fig.text(.7,.1,"An empty subplot means that all blinks \n were between saccades for that run")
     
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/non_surrounded_blinks_s{subject_num}_r{run_num}.png")
    plt.close(fig)

#%%
print("number of blinks with surrounding saccades for all subs and runs")
total=0
for item in all_sub_run_sur_saccs:
    total += sum(item)
print(total)

print("number of blinks total for all subjects and runs")
print(len(all_blinks))

# even if we allow 100ms netween blink and saccade, there are thousands of blinks w/o saccades around
# 1s buffer almost all blinks are surrounded by saccs, but this seems quite large

#%%
print(all_blinks["eye"].isna().sum())
#%%
"""
def is_blink_between_saccades(blink_row, saccades_df):
    # Filter saccades by the same subject and run
    saccades_subset = saccades_df[(saccades_df['Subject'] == blink_row['Subject']) & 
                                  (saccades_df['run_num'] == blink_row['run_num'])]
    
    # Find the saccade that ends right before the blink starts
    saccade_before = saccades_subset[saccades_subset['tEnd'] < blink_row['tStart']].tail(1)
    
    # Find the saccade that starts right after the blink ends
    saccade_after = saccades_subset[saccades_subset['tStart'] > blink_row['tEnd']].head(1)
    
    # Check if both saccades are found (i.e., not empty) 
    if not saccade_before.empty and not saccade_after.empty:
        return True
    else:
        return False

# Apply the function to each blink in the blinks dataframe
sorted_blinks['between_saccades'] = sorted_blinks.apply(lambda row: is_blink_between_saccades(row, sorted_saccades), axis=1)

# Now, check if all blinks are between two saccades for each subject and run
result_by_subject_run = sorted_blinks.groupby(['Subject', 'run_num'])['between_saccades'].all()

# Print the result
if result_by_subject_run.all():
    print("For all subjects and runs, all blinks are between two saccades.")
else:
    print("There are some subjects and runs where not all blinks are between two saccades.")
    blinks_between_saccades = sorted_blinks["between_saccades"].sum()
    blinks_not_between_saccades = len(sorted_blinks) - blinks_between_saccades
    print(f"Number of blinks between two saccades: {blinks_between_saccades}")
    print(f"Number of blinks not between two saccades: {blinks_not_between_saccades}")

"""


#%%
# get info about data 
chunks = []
for chunk in pd.read_csv("/Volumes/brainlab/Mindless Reading/neuralnet_classifier/all_subjects_interpolated_pupil_coord.csv", chunksize=100000):
    chunks.append(chunk)
    
dfSamples = pd.concat(chunks,ignore_index=True)
dfSamples.drop(subset="Unnamed: 0", axis=1, inplace=True)

#%%
#%% CURRENT STARTS HERE 10/21
# no forward fill, use saccades, no s10127

subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]

# pupil interpolation with within-subject normalization first
for subject_num in subjects:
    for run_num in range(1,6):
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_sample_filenames = [file for file in filenames if (file.endswith("Sample.csv") and not file.startswith("._"))]
        sample_file = matching_sample_filenames[0]
        print("sample file", sample_file)
        matching_blink_filenames = [file for file in filenames if (file.endswith("Blink.csv") and not file.startswith("._"))]
        blink_file = matching_blink_filenames[0]
        print("blink file", blink_file)
        dfSamples = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{sample_file}")
        dfSamples = normalize_pupil_pre_interp(dfSamples)
        dfBlink = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{blink_file}")
        matching_saccade_filenames = [file for file in filenames if (file.endswith("Saccade.csv") and not file.startswith("._"))]
        saccade_file = matching_saccade_filenames[0]
        print("saccade file", saccade_file)
        dfSaccades = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{saccade_file}")
        # coordinate interpolation
        dfSamples = interpolate_coordinates_over_blinksv2_nomerge(dfSamples, dfBlink, dfSaccades, subject_str, run_num)
        # pupil interpolation
        dfSamples = interpolate_pupil_over_blinksv2_nomerge(dfSamples, dfBlink, dfSaccades, subject_str, run_num)
        print("LPupil min")
        print(dfSamples["LPupil_normalized"].min())
        print("RPupil min")
        print(dfSamples["RPupil_normalized"].min())
        
"""
# coordinate interpolation without forward fill first
for subject_num in subjects:
    for run_num in range(1,6):
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_blink_filenames = [file for file in filenames if (file.endswith("Blink.csv") and not file.startswith("._"))]
        blink_file = matching_blink_filenames[0]
        print("blink file", blink_file)
        matching_saccade_filenames = [file for file in filenames if (file.endswith("Saccade.csv") and not file.startswith("._"))]
        saccade_file = matching_saccade_filenames[0]
        print("saccade file", saccade_file)
        # feed in file that already had pupil interpolation
        dfSamples = pd.read_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_str[-3:]}_r{run_num}_Sample_Interpolated.csv")
        dfBlink = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{blink_file}")

        # forward fill missing coordinates
        #dfSamples[["LX", "LY", "RX", "RY"]] = dfSamples[["LX", "LY", "RX", "RY"]].fillna(method="ffill")

        # interpolate
        interpolate_coordinates_over_blinksv2(dfSamples, dfBlink, dfSaccades, subject_str, run_num)
"""
# extract interpolated from
# /Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_num[-3:]}_r{run}_Sample_Interpolated.csv
for subject in subjects:
    folder_path = f"/Volumes/Lexar/CSV_Samples/{subject}"
    extract(subject, folder_path)
    
# now files are at /Volumes/Lexar/MW_Classifier_Data/{sub_id}_raw.csv
    
#%%%
# start here to save csv no need to regenerate plots
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
data_path = "/Volumes/Lexar/MW_Classifier_Data/"
dfSamples = load_raw_samples(data_path, subjects)
dfSamples = preprocess(dfSamples)

#%%
print(dfSamples.columns)


print(dfSamples["Subject"])


print(dfSamples["LPupil_normalized"].min())
print(dfSamples["RPupil_normalized"].min())

# drop cols if needed
#dfSamples = dfSamples.drop(["Unnamed: 0", "LPupil", "RPupil"], axis=1)

# consider replacing 0s with na and forward filling? prior to interpolation
# both of these are 0 :( 

# plots
print("creating corr heatmap")
correlation_heatmap(dfSamples)
print("freq by subject")
mw_freq_by_subject(dfSamples)
print("mw over time")
mw_over_time(dfSamples)
print("pupil over time")
pupil_over_time(dfSamples)
print("pupil subject run")
pupil_subject_run(dfSamples, subjects)
print("eye coords")
eye_coordinates(dfSamples, subjects)
print("coord over time")
coord_over_time(dfSamples)

#%% check for missing values

print(dfSamples.isna().sum())
print(dfSamples.columns)
#%%
# drop rows with missing coordinate values
dfSamples = dfSamples.dropna()
#%% 
# drop unnecessary cols
#dfSamples = dfSamples.drop(columns=["LPupil", "RPupil", "Unnamed: 0"])
# save to csv in netfiles
#dfSamples.to_csv("/Volumes/brainlab/Mindless Reading/neuralnet_classifier/all_subjects_interpolated.csv")
# and lexar
dfSamples.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_subjects_interpolated.csv")
#%% 
print(dfSamples["Subject"])

#%% 
# read csv
chunks = []
chunk_count = 0
total_rows = 0
chunksize = 500000
for chunk in pd.read_csv("/Volumes/Lexar/MW_Classifier_Data/all_subjects_interpolated_pupil_coord.csv", chunksize=chunksize):
    chunks.append(chunk)
    chunk_count += 1
    total_rows += len(chunk)
    # print progress
    print(f"Processed chunk {chunk_count}, total rows processed: {total_rows}")
# concat chunks into df
dfSamples = pd.concat(chunks,ignore_index=True)
#%% Current ends here

dfSamples["Subject"] = dfSamples["Subject"].astype(str)
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
print("creating corr heatmap")
correlation_heatmap(dfSamples)
print("freq by subject")
mw_freq_by_subject(dfSamples)
print("mw over time")
mw_over_time(dfSamples)
print("pupil over time")
pupil_over_time(dfSamples)
print("pupil subject run")
pupil_subject_run(dfSamples, subjects)
print("eye coords")
eye_coordinates(dfSamples, subjects)
print("coord over time")
coord_over_time(dfSamples)


#%% Create uninterpolated with blink cols
# add blink col to csv sample for each subject and run
    # have csv sample files end in "Sample_with_blink.csv" after this

subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]


for subject_num in subjects:
    for run_num in range(1,6):
        # load in samples and blinks for this sub and run
        subject_str = str(subject_num)
        foldernames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/")
        matching_folders = [folder for folder in foldernames if (folder.startswith(f"s{subject_str[-3:]}_r{run_num}") & folder.endswith("data"))]
        folder = matching_folders[0]
        filenames = os.listdir(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}")
        matching_sample_filenames = [file for file in filenames if (file.endswith("Sample.csv") and not file.startswith("._"))]
        sample_file = matching_sample_filenames[0]
        print("sample file", sample_file)
        matching_blink_filenames = [file for file in filenames if (file.endswith("Blink.csv") and not file.startswith("._"))]
        blink_file = matching_blink_filenames[0]
        print("blink file", blink_file)
        dfSamples = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{sample_file}")
        dfBlink = pd.read_csv(f"/Volumes/brainlab/Mindless Reading/DataCollection/s{subject_num}/eye/{folder}/{blink_file}")
        # add Lblink and Rblink to dfSamples
        # if tSample is within tStart and tEnd for a row of blink and "eye" matches L or R, col is 1 at that index
        # initialize new cols to hold 0s
        dfSamples["Rblink"] = 0
        dfSamples["Lblink"] = 0
        # loop through rows of dfBlink
        for index, row in dfBlink.iterrows():
            # get blink start, end, and eye
            eye = row["eye"]
            tStart = row["tStart"]
            tEnd = row["tEnd"]
            # create mask for samples within blink duration
            mask = (dfSamples["tSample"] >= tStart) & (dfSamples["tSample"] <= tEnd)
            # change value in appropriate blink col for indices in mask based on eye value
            if eye == "L":
                dfSamples.loc[mask, "Lblink"] = 1
            elif eye == "R":
                dfSamples.loc[mask, "Rblink"] = 1
        # save dfSamples to new csv 
        dfSamples.to_csv(f"/Volumes/Lexar/CSV_Samples/{subject_num}/s{subject_str[-3:]}_r{run_num}_Sample_with_blink.csv")
        
#%%

# extract samples
for subject in subjects:
    folder_path = f"/Volumes/Lexar/CSV_Samples/{subject}"
    extract(subject, folder_path)
#%%

one_sub = pd.read_csv("/Volumes/Lexar/CSV_Samples/10014/s014_r1_Sample_with_blink.csv")
print(one_sub.dtypes)
#%%
subjects = [10014,10052,10059,10073,10080,10081,10084,10085,10089,10092,10094,
            10100,10103,10110,10111,10115,10117,10121,10125]
# load and preprocess - make sure this includes the pupil normalization
data_path = "/Volumes/Lexar/MW_Classifier_Data/"
dfSamples = load_raw_samples(data_path, subjects)
dfSamples = preprocess(dfSamples)
print(dfSamples.columns)
#%%
# drop cols if needed
dfSamples = dfSamples.drop(["Unnamed: 0", "LPupil", "RPupil"], axis=1)
# forward fill missing coordinates
dfSamples[["LX", "LY", "RX", "RY"]] = dfSamples[["LX", "LY", "RX", "RY"]].fillna(method="ffill")
# save csv
print("saving to csv...")
dfSamples.to_csv("/Volumes/Lexar/MW_Classifier_Data/all_sub_no_interp_with_blink.csv")
# plot, change corr heatmap to include R and Lblink first
print("creating corr heatmap")
correlation_heatmap(dfSamples)
print("freq by subject")
mw_freq_by_subject(dfSamples)
print("mw over time")
mw_over_time(dfSamples)
print("pupil over time")
pupil_over_time(dfSamples)
print("pupil subject run")
pupil_subject_run(dfSamples, subjects)
print("eye coords")
eye_coordinates(dfSamples, subjects)
print("coord over time")
coord_over_time(dfSamples)

#%%
# rerun fntn then gen corr heatmap, pupil per subject run, eye coords, and coord over time
dfSamples["run_num"] = dfSamples["run_num"].astype(float)
#%%
print("creating corr heatmap")
correlation_heatmap(dfSamples)
print("pupil over time")
pupil_over_time(dfSamples)
print("pupil subject run")
pupil_subject_run(dfSamples, subjects)
print("eye coords")
eye_coordinates(dfSamples, subjects)
print("coord over time")
coord_over_time(dfSamples)
