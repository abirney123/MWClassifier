# ParseEyeLinkAsc_script.ipynp
#
# Created 8/15/18 by DJ.
# Modified 10/26/22 by HS -- Update image dimensions
#                         -- Implement a dataframe in words_of_pages
# Modified 10/10/23 by HS - update words_of_pages function
# Updated on 10/31/23 by HS - directly read in eye features if they have alraedy
#                           - been parsed and saved
# Updated on 2/13/24 by HS - use eye samples for pupil analyses
# Updated 4/16/24 by HS - interpolate pupil size during blink

# Import packages
import os
import time
import string # to remove punctuation to find zipf scores.
import numpy as np
import pandas as pd
from parse_EyeLink_asc import ParseEyeLinkAsc
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from page import Page
from textblob import TextBlob
from scipy.interpolate import CubicSpline

# DONE: Function(s) for image pixel mapping @Oumou
# DONE: Mean Pupil Size @Blythe
# DONE: Rewrite Main function to get and process file information @George
# DONE: Add Error image parsing to the img_page_data array based on error @George
# DOne: Add error word discovery based on csv file and add that error word if fixated on @George
# DONE: Update eyelink message output to show correct error page @Haorui
# Done: Organize dictionary based on words on the page and add fixation counts, pupil and other features to that @George
# Done: Plot the CSV file and make it look nice over the course of the experiment @George
# TODO: Calculate % of time not looking at words (fixations not on words)


def get_data(elFilename, dataDir, is_overwrite=False):
    '''
    input:
        filename: name of file for eyelink file
    output: extracted data from parsing asc eyelink file
    '''

    # Declare filenames
    elFileStart = os.path.splitext(elFilename)[0]
    # dataDir = "/home/george/Documents/GitHub/GlassBrain/reading-analysis/Examples" # folder where the data sits
    # elFilename = 'G_Run1_2022_03_03_11_08.asc' # filename of the EyeLink file (.asc)
    outDir = f'{dataDir}/{elFileStart}_data'
    
    # make the directory if not exists
    is_outDir_exists = os.path.exists(outDir)
    if not is_outDir_exists:
        os.mkdir(outDir)
        is_overwrite = True
    # check the folder contains all parsed files
    # reparse all eye features if missing any file
    else:
        if len(os.listdir(outDir)) != 6:
            is_overwrite = True
            
            
    # Make master list of dataframes to write (or read)
    allDataFrames = ['_']*6 # see following code about the dataframes
    allNames = ['Trial','Message','Fixation','Saccade','Blink','Sample'] # what they're called
    
    # parse eye features if overwrite is true
    if is_overwrite:
        print('Parse eye data and overwrite any existing files...')
        # Load file in
        dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(f'{dataDir}/{elFilename}')
        print('Saving results...')
    
        # Make master list of dataframes to write
        allDataFrames = [dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples] # the dataframes
        allNames = ['Trial','Message','Fixation','Saccade','Blink','Sample'] # what they're called
    
        # Write dataframes to .csv files
        for i in range(len(allNames)):
            # save csv files
            outFilename = '%s/%s_%s.csv'%(outDir,elFileStart,allNames[i])
            print('   Saving %s output as %s...'%(allNames[i],outFilename))
            allDataFrames[i].to_csv(outFilename,float_format='%.1f',index=False)
    
    # else read in existing .csv files
    else:
        print('Directly read in parsed csv files...')
        for i in range(len(allNames)):
            # read in csv files
            outFilename = '%s/%s_%s.csv'%(outDir,elFileStart,allNames[i])
            allDataFrames[i] = pd.read_csv(outFilename)
            
        print('Done!')
    return allDataFrames
    #return dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples



######################################################
##### Parse Page Number by Timestamp in Message ######
######################################################

def page_timestamp(dfMsg, tau):
    '''
    intput: dfMsg - pandas dataframe of mesages from eyelink
    output: page_stamps - list of timestamps for each page start

    dfMesg has the following types of messages that we care about:
        - "TRIALID xx" where "xx" is the current page being displayed
        - "displayed error image xx" where "xx" is the image being displayed, this is the
            part of the experiment where the user is clicking on an error
        - "displayed image xx" where "xx" is the image being displayed, this is the part
            where the participant is actively doing the reading of the page
        - "display thought probe" which indicates the participant is doing a thought probe
        - "start error routine" which indicates the user is selecing the error and answering
            thought probe questions
    '''

    # DONE:
    # Make sure the timestamps are for just the pages the participant was reading the page
    # Do not include the looking for error pages or the thought probe pages
    # As of now (04/2022) the eyelink messages include the "looking for error" page. Not good

    # UPDATE: As of 05/16 this works now.
    # Now the timestamps divide the pages everytime the user is required to read a page. This
    # happens during the 1st pass through a reading and when the user has to select an error
    # on the page. Each page will know what state it is in. See the `Page.py` file for more information
    
    pages = []
    page_end_search = False # look for the end time of the page
    page_complete = False # look for the end time of the page
    page_view = '1st Pass'
    page_reference = 'None'
    pages_time = dfMsg['time'].values
    pages_page = dfMsg['text'].values
    for index, message in enumerate(pages_page):
        
        # looking for the timestamp when the task begins
        if 'Reading START' in message:
            task_start = pages_time[index]/1000

        # looking for image file. This is our new page start
        if '.png' in message: # this means an image is being viewed

            page_view = '1st Pass'
            message_value = message.split(' ')
            page_reference = message_value[-1]
            if ('errorimage' in message): # this means it's an error image or MW selection page
                page_view = 'Select Error' # where participant selects MW start & end words
            elif('RE: displayed' in message): # repeat page after reporting MW
                page_view = '2nd Pass'
            # page_reference is imageLocation
            # eg. the_voynich_manuscript/the_voynich_manuscript_control/
            # the_voynich_manuscript_control_page01.png
            new_page = Page(pages_time[index]/1000, page_reference, page_view) # convert time to seconds

            # look for the end time of the page
            page_end_search = True

        # now we're looking for the end time, so grab the next timestamp
        elif (('Current Page END' in message) and page_end_search):
            new_page.time_end = pages_time[index]/1000 # convert to seconds
            
            # use time window instead of the whole page
            if tau:
                new_page.time_start = max(new_page.time_end-tau, new_page.time_start) #EyeLink fs: 2000 Hz
                
            new_page.calculate_duration() # calculate page duration in seconds
            # Now just start looking for first page again
            # this also flags system to append page to list
            page_end_search = False
            page_complete = True
            new_page.task_start = task_start

        # this means start and end times were found. append new page
        if (page_complete):           
            pages.append(new_page)
            page_complete = False

    return pages



######################################################
###### sort information based on time stamps #########
######################################################

def sort_times(dfBlink, dfFix, dfSacc, dfSamples, pages, eye):
    '''
    input:  dfBlink - blinks of data
            dfFix - fixations of data
            dfSacc - saccades of data
            page_stamps - timstamps for start of pages
    output: pages_data - dictionary of data broken down by pages
    '''
    # add a new column to the dataframe to specify valid time window
    # NOTE that we need to use original dataframe for this instead of using the
    # copy from the for-loop later
    dfBlink['in_win'] = 1
    dfFix['in_win'] = 1
    dfSacc['in_win'] = 1

    # extract single eye data
    if (eye!='mono'):
        indices = dfBlink['eye'] == eye
        dfBlink = dfBlink.iloc[indices.to_list()]
        
        indices = dfFix['eye'] == eye
        dfFix = dfFix.iloc[indices.to_list()]
        
        indices = dfSacc['eye'] == eye
        dfSacc = dfSacc.iloc[indices.to_list()]
        
        dfSamples = dfSamples[['tSample', f'{eye}X', f'{eye}Y', f'{eye}Pupil']].copy()
        # rename the pupil column
        dfSamples.rename(columns={f'{eye}Pupil':'pupil'}, inplace=True)
    else:
        # rename the pupil column
        if len(dfSamples['LPupil'])==0:
            dfSamples.rename(columns={'RPupil':'pupil'}, inplace=True)
        else:
            dfSamples.rename(columns={'LPupil':'pupil'}, inplace=True)
    
    
    # interpolate the pupil size during the blink duration
    # http://dx.doi.org/10.6084/m9.figshare.688002
    # get the time of every sample
    sample_time = dfSamples['tSample'].to_numpy()
    pupil = np.array(dfSamples['pupil'])
    # declare blink offset: 50ms added to the start and end of a blink
    blink_off = 50
    
    # declare variables to store blink start information to merge blinks if 
    # they are too close 
    update_start = True
    # iterate throu each row of blink dataframe
    for index in np.arange(len(dfBlink)):
        row = dfBlink.iloc[index]
        # get the start and end time
        cb_start = row['tStart'] - blink_off
        b_end = row['tEnd'] + blink_off
        # update b_start if necessary
        if update_start:
            b_start = cb_start
            
        if index+1 < len(dfBlink):
            # get next blink sample
            nrow = dfBlink.iloc[index+1]
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
            # select time and pupil size for interpolation
            x = [t1, t2, t3, t4]
            y_ind = []
            for t in x:
                y_ind.append(np.where(sample_time==t)[0][0])
            y = pupil[y_ind]
            # generate the spl model using the time and pupil size
            spl = CubicSpline(x, y)
            
            # generate mask for blink duration
            mask = (sample_time > t2) & (sample_time < t3)
            x = sample_time[mask]
            # use spl model to interpolate pupil size for blink duration
            interp_pupil = spl(x)
            
            # update the dfSamples in place
            dfSamples.loc[mask, 'pupil'] = interp_pupil
        
    
    # pupil1 = np.array(dfSamples['pupil'])
    # plt.figure()
    # plt.plot(pupil)
    # plt.plot(pupil1, '.')
    
    # downsample the samples dataframe
    # original sampling frequency is 1000 Hz. We downsample it to 100 Hz to 
    # speed up pupil analyses
    downsample_factor = 10
    dfSamples = dfSamples.iloc[::downsample_factor, :]
    # add a new column to the sample dataframe now
    dfSamples['in_win'] = 1
    
    # pupil2 = dfSamples['pupil'].to_numpy()
    # plt.figure()
    # t = np.arange(len(pupil))/1000
    # t1 = np.arange(len(pupil2))/100
    # plt.plot(t, pupil)
    # plt.plot(t1, pupil2, '.')
    
    # loop through each page object
    # select features for the current page
    # convert start and end time back to miliseconds for comparison
    # return the whole dataframe without the first column (eye)
    for page in pages:
        # blinks
        blink_indices = (dfBlink['tStart'] > page.time_start*1000) & \
                        (dfBlink['tEnd'] < page.time_end*1000)
        page.blinks = dfBlink.loc[blink_indices].drop(columns=['eye'])
        # fixatoins
        fix_indices = (dfFix['tStart'] > page.time_start*1000) & \
                      (dfFix['tEnd'] < page.time_end*1000)
        page.fixations = dfFix.loc[fix_indices].drop(columns=['eye'])
        # saccades
        sacc_indices = (dfSacc['tStart'] > page.time_start*1000) & \
                       (dfSacc['tEnd'] < page.time_end*1000)
        page.saccades = dfSacc.loc[sacc_indices].drop(columns=['eye'])
        # eye samples (pupil info)
        pupil_indices = (dfSamples['tSample'] > page.time_start*1000) & \
                        (dfSamples['tSample'] < page.time_end*1000)
        page.pupils = dfSamples.loc[pupil_indices]
            
         
        
def convert_pixel_to_eyelink(x_image,y_image):
    """
    converts image pixels  to eyelink pixels

    inputs:

    x_image: float
    The term to convert x coordinates to eye link

    y_image: float
    The term to convert y coordinates to eyelink
​
    output:

    x_eyelink : float
    the converted x coordinate

    y_eyelink: float
    the converted y coordinate

    (Assuming we are converting from the dimensions 1209x918 to 1418x1070)(x,y)
    The dimensions for updated images: 1900x1440
    """

    #changes x to eyelink coordinates
    # x_eyelink = (1418/1209)*x_image+355.2
    x_eyelink = (1080 * 1.3/1900)*x_image+258

    #changes y to eyelink coordinates
    # y_eyelink = (1070/918)*y_image+27
    y_eyelink = (1080 * 0.99/1442)*y_image+5.4

    return x_eyelink,y_eyelink


def convert_py_to_eyelink( x_psycho , y_psycho ):
    """
    Converts py to eyelink terms

    Inputs:

    psycho_x float
    The term to convert psychopy x-coordinates

    psycho_y: float
    The term the psychopy y-coordinates

    output:

    x_pixel: float
    The converted x coordinate [py to eyelink]

    y_pixel: float
    the converted y coordinate [py to eyelink]

        (Assuming we are converting from the dimensions 1.12x0.85 to 1418x1070 (x,y)
    11/07/22 Updated: 1.3 x 0.99 to 1900x1442
     """


    #changes x to from psychopy to eyelink coordinates
    x_eyelink = (1209/1.3)*x_psycho+960

    #changes y to from psychopy to eyelink coordinates
    y_eyelink = -(918/0.99)*y_psycho+540

    return  x_eyelink, y_eyelink

def convert_pixel_to_py(x_image, y_image):
    '''
    Convert image pixel to psychopy height unit

    Parameters
    ----------
    x_image : float
        DESCRIPTION. position x in pixel
    y_image : float
        DESCRIPTION. position y in pixel

    Returns
    -------
    x_py : float
        DESCRIPTION. position x in height unit
    y_py : float
        DESCRIPTION. position y in height unit

    '''
    x_py = (x_image-1900/2)/1900 * 1.3
    y_py = (y_image-1440/2)/1440 * 0.99
    
    return x_py, y_py

######################################################
###### Get Information from CSV file for reading #########
######################################################
def create_zipf_dict(zipf_filename = 'word_sensitivity_table.xlsx'):
    return pd.read_excel(zipf_filename, usecols=['Word', 'FreqZipfUS']).set_index('Word').to_dict()['FreqZipfUS']


def words_of_pages(reading_dir, page, zipf_dict=None):
    '''
    inputs: filename - csv of story words
    output: moimg_page_data - each page with word positions of the story
    '''

    # make the file name using the Page reference
    image_path = page.image_file_location.split('/')
    filename = f'{reading_dir}/{image_path[0]}/{image_path[1]}/{image_path[1]}_coordinates.csv'

    # read in the coordinate file as pd dataframe
    coordinate_df = pd.read_csv(filename)
    page_specific_df = coordinate_df[coordinate_df['page'] == page.page_number]
    
    # drop unnecessary columns
    page_specific_df = page_specific_df.drop(columns=['left', 'top', 'prop_height', 'prop_width'])
    
    # add new columns (fixation_duration, word_fixation_times, zipf_scores) to words pd
    # and set default values to 0
    for new_col in ['word_fix_times', 'fix_duration', 'zipf']:
        page_specific_df[new_col] = np.nan # nan to avoid biasing correlation and to make it a float
    
    # loop through each row to find the zipf score
    for row_index, row in page_specific_df.iterrows():
        word = row['words'].lower() # convert to lower-case
        word.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        # use try-except to ignore words cannot be recognized by textblob
        try:
            word = TextBlob(word).words[0].singularize() # remove plurals
        except:
            pass
            
        if zipf_dict and word in zipf_dict.keys():
             page_specific_df.at[row_index, 'zipf'] = zipf_dict[word]

    # true_words is the dataframe
    page.true_words = page_specific_df.reset_index()

            

######################################################
###### Match fixations to the word it's on #########
######################################################

# Moved all of this to main.py


#######################
##### STATISTICS ######
#######################

def page_statistics(pages_data, feature_analysis):
    '''
    input:  pages_data - dictionary of data broken down by pages
            features_to_analyze - array of strings for each feature to analyze
    output:
            feature_count_data - Count of events per page of feature in list of features
            feature_average_data -
    '''
    feature_count_data = {}
    feature_average_data = {}

    # Count the number of actions per page
    num_actions_per_page = 0
    for feature in feature_analysis:
        feature_count_data[feature] = []
        for page in range(len(pages_data[feature])):
            num_actions_per_page = np.count_nonzero(pages_data[feature][page][0])

            feature_count_data[feature].append(num_actions_per_page)

    return feature_count_data, feature_average_data


#############################
##### Figures of Merit ######
#############################

def no_animate(file_location, page_data, ax, eye=None):
    '''
    input:  file_location - location of the image (from Page object)
            page_data - the page object of interest
            ax - which axis to plot the data on
    output: None, show's plot, close plot to show next image (if available)
    '''
    im = plt.imread(file_location) # get coresponding page image index
    

    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4)) # hardcode based on screen / image size
    ax.set_xlim([0,1920]) # fix the y axis to the size of the screen display
    ax.set_ylim([0, 1080]) # fix the y axis to the size of the screen display
    ax.set_ylim(ax.get_ylim()[::-1])
    
    fix_data = page_data.feature_data['fix']
    # gaze pos x
    x_data = np.array(fix_data['xAvg'])
    # gaze pos y
    y_data = np.array(fix_data['yAvg'])
    # pupil size
    data_size = np.array(fix_data['duration'])/3
    
    # scatter plot the fixations as dots
    ax.scatter(x_data, y_data, c=np.arange(1,len(x_data)+1), s=data_size, alpha=0.7)
    
    # differentiate fixations within the reported time window
    if page_data.mw_reported:
        fix_mask = np.array(fix_data['in_win'], dtype=bool)
        # scatter plot the fixations as dots
        ax.scatter(x_data[fix_mask], y_data[fix_mask], c='r', 
                   s=data_size[fix_mask]/2, alpha=0.9)
    
    # plot a line between two fixations
    for index in np.arange(1, len(x_data)):
        x = [x_data[index-1], x_data[index]]
        y = [y_data[index-1], y_data[index]]
        ax.plot(x, y, 'r', alpha=0.3)
    

    # rectangles to show the bouding box of each word
    word_color = '#CBE4ED'
    for _, true_word in page_data.true_words.iterrows():

        # define the bounding box of the word
        word_x_start_img_pix = true_word['center_x'] - (true_word['width'] / 2)
        word_x_end_img_pix = true_word['center_x'] + (true_word['width'] / 2)
        word_y_start_img_pix = true_word['center_y'] - (true_word['height'] / 2)
        word_y_end_img_pix = true_word['center_y'] + (true_word['height'] / 2)

        # Convert image word pixels to eyelink pixels
        word_x_start,word_y_start = convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
        word_x_end, word_y_end = convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)

        # update color of highlight
        if (true_word['is_error']): # if is_error
            word_color = "#f77959" # red-ish color
            # just plot the clicked word
            ax.add_patch(plt.Rectangle((word_x_start, word_y_start), word_x_end-word_x_start, word_y_end-word_y_start, alpha=0.5, color=word_color))
        else:
            word_color = '#CBE4ED' # light grey

        #ax.add_patch(plt.Rectangle((word_x_start, word_y_start), word_x_end-word_x_start, word_y_end-word_y_start, alpha=0.2, color=word_color))

    ax.set_title(f'Page Index {page_data.page_number} - Fixations with Duration - {page_data.page_view} - {page_data.error_type} - Eye {eye}')

def plot_norm_data(pages_data, ax):
    '''
    Plot nomalized data for all pages in the series.
    input:  pages_data - array of Page objects
            ax - axis to plot the data
    output  None. Creates plot
    '''

    # initialize array to store the page data
    num_fix_pages = []
    pupil_slopes_pages = []
    num_blinks_pages = []
    num_saccades_pages = []

    # add each page data to the array
    for page in pages_data:
        num_fix_pages.append(page.num_fixations)
        pupil_slopes_pages.append(page.pupil_slope)
        num_blinks_pages.append(page.num_blinks)
        num_saccades_pages.append(page.num_saccades)

    # plot the data
    ax.plot(np.arange(0,len(pages_data)), num_fix_pages, c='r', label='fixations')
    ax.plot(np.arange(0,len(pages_data)), pupil_slopes_pages, c='b', label='pupil slope')
    ax.plot(np.arange(0,len(pages_data)), num_blinks_pages, c='m', label='blinks')
    ax.plot(np.arange(0,len(pages_data)), num_saccades_pages, c='k', label='saccades')

    # format the plot
    ax.set_title("Normalized Data for All Pages")
    ax.legend()


def animate(i, ax, page_index, file_location, pages_data):
    '''
    input: i - the index of the data to show
            pages_data - dictionary of data broken down by pages
            page_index - the page to plot
    output: None, show's plot, close plot to show next image (if available)
    '''
    im = plt.imread(file_location)
    ax.cla() # clear the previous image
    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4))
    

    ax.set_xlim([0,1920]) # fix the y axis
    ax.set_ylim([0, 1080]) # fix the y axis
    ax.set_ylim(ax.get_ylim()[::-1])
    x_data = np.array(pages_data[page_index].feature_data['fix'][0])[:i]
    y_data = np.array(pages_data[page_index].feature_data['fix'][1])[:i]
    # data_size = np.array(pages_data[page_index].feature_data['fix'][2])
    # ax.scatter(pages_data[page_index].feature_data['fix'][0]-x_offset, pages_data[page_index].feature_data['fix'][1]-y_offset) # plot the line
    # ax.scatter(x_data, y_data, c=np.arange(1,len(pages_data[page_index].feature_data['fix'][0])+1), s = data_size) # plot the line
    ax.scatter(x_data, y_data, c=np.arange(1,len(pages_data[page_index].feature_data['fix'][0])+1)[:i]) # plot the line

    # format the plot
    ax.set_title(f'Page Index {page_index} Fixations')
    
    
def find_match(words_info, click_pos, dist_max=1000):
    '''
    Match the clicks to words. 

    Parameters
    ----------
    words_info : DataFrame
        DESCRIPTION. The dataframe for a single page. It should at least contain
        columns 'center_x', 'center_y', 'width', and 'height', which are 
        coordinate information of words in pixel unit
    click_pos : Tuple
        DESCRIPTION. The coordinate info (x, y) for a click in pixel unit
    dist_max : float
        DESCRIPTION. The threshold value to accept minimum distance
        Default: 1000

    Returns
    -------
    matched_index: int
        DESCRIPTION. The index value of matched word for input dataframe. 

    '''
    # compute the start and end positions of words
    words_x_start = words_info['center_x'] - words_info['width']/2
    words_x_end = words_info['center_x'] + words_info['width']/2
    words_y_start = words_info['center_y'] - words_info['height']/2
    words_y_end = words_info['center_y'] + words_info['height']/2
    
    # get x and y for the click
    click_x, click_y = click_pos
    
    # compute the distance between click and word boundry box
    dist_x_left = (words_x_start - click_x)
    dist_x_right = (click_x - words_x_end)
    dist_y_top = (words_y_start - click_y)
    dist_y_bottom = (click_y - words_y_end)
    
    # find the maximum distance from click to the word for x and y
    max_x = np.max(np.vstack((dist_x_left, dist_x_right, np.zeros(len(dist_x_left)))), axis=0)
    max_y = np.max(np.vstack((dist_y_top, dist_y_bottom, np.zeros(len(dist_y_top)))), axis=0)
    
    # calculate the distance using x and y
    dist = np.sqrt(np.square(max_x) + np.square(max_y))
    
    # check if the minimum dist exceeds threshold values
    if np.min(dist) < dist_max:
        matched_index = np.argmin(dist)
    else:
        matched_index = -1
    
    # return the index that has the shortest distance
    return matched_index
    

    
