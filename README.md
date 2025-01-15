# Mind Wandering Classifier

This repo contains files related to the deep learning mind wandering classifier. In this project, we design and implement a deep learning system to classify whether or not subjects were experiencing a subconscious drift of attention away from the task at hand (Mind Wandering). Our final system consists of a combined CNN-LSTM architecture with an attention mechanism. Eye tracking data from subjects for which data was collected by the University of Vermont's Glass Brain Lab was used for this model. The files used for this model and their functions are listed below (please note that the GitHub repository contains various other files that were used in the development process, but may be less relevant to readers at this time):
1. Load_Preprocess_EDA.py: A script for loading and preprocessing raw data as well as performing exploratory data analysis on data after preprocessing.
2. Reading_analysis.py: The original code from which the data interpolation functions were derived from.
3. parse_eyeLink_asc.py: Combines raw, unprocessed eye-tracking data with metadata about the experiment (such as whether or not MW was reported to be occurring during each timepoint) to result in labeled samples for each millisecond of recorded time. This is referenced in the Load_Preprocess_EDA script.
4. Balance_data.py: The script used to perform selective downsampling as described in this report.
5. prep_data_LSTM.py: The script used to perform the train test split as described in this report.
6. train_CNN-LSTM.py: The training script for the CNN-LSTM as described in the section “Deep
Learning System”.
7. test_CNNLSTM.py: The testing script for the CNN-LSTM as described in the section “Deep
Learning System”.
8. train_CNN-LSTM_experiment.py: The training script for the CNN-LSTM as described in the
section “CNN Architecture Modification”.
9. test_CNN-LSTM_experiment.py: The testing script for the CNN-LSTM as described in the
section “CNN Architecture Modification”.
10. train_LSTM_new_windows.py: The training script for the LSTM without the CNN feature
extractor.
11. test_LSTM_with_analysis_eff.py: The testing script for the LSTM without the CNN feature extractor. Contains functionality to aggregate classifications for timesteps corresponding to the same subject, run, and page, repeated in multiple overlapping windows, plot errors over time for each test subject, and plot a heatmap of the confusion matrix.
12. CNN.py: the program for training and evaluating the CNN model, sliding window mechanism for sequence level labeling.
13. CNN_ULSTM.py: the program for training and evaluating the unidirectional CNN-LSTM model, sliding window mechanism for sequence level labeling.
14. train_CNN_LSTM_Att.py: the program for training the directional CNN-LSTM model extended with the use of attention mechanisms on both the CNN and LSTM models.
15. test_CNN_LSTM_Att.py: the program for evaluating the directional CNN-LSTM model extended with the use of attention mechanisms on both the CNN and LSTM models.

A full report regarding this project can be found within the file "Mind_Wandering_Deep_Learning_Classifier_Extended": 
