#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:53:46 2024

@author: Alaina
Use TorchGPU conda environment 


"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1Score, AUROC, MatthewsCorrCoef
from torcheval.metrics import BinaryAUROC
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sn
#from torchvision.ops import sigmoid_focal_loss

#%%


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
        dfSamples.drop("Unnamed: 0", axis=1, inplace=True)
    return dfSamples


def aggregate_probabilities(labels, probabilities, sequence_lengths, start_timesteps, threshold=0.5):
    """
    Aggregate probabilities from multiple windows that cover the same
    timesteps to provide more robust classifications. Additionally, aggregate
    labels so they are comparable.
    Accepts:
        - labels: List of arrays. Truth labels across all timesteps, flattened into a single
        tensor. 
        - probabilities: List of arrays. Predicted probabilities for each window.
        Each probability must correspond to the probability generated for a 
        certain window that starts at start_timestep and continues for sequence_length
        timesteps.
        - sequence-lengths: List of int. Sequence lengths for each window.
        - start_timesteps: List of int. Start timesteps for each window.
        - threshold: Float.Threshold for converting probabilities to classes. If the mean
        probability is over or equal to this threshold, the classification will be
        positive (MW). Optional, default is .5.
    Returns:
        - classifications: Dictionary. Aggregated predictions. Each key is a timestep,
        each value is the class.
        - agg_probabilities: Dictionary. Aggregated probabilities. Each key is a timestep,
        each value is the probability of the classification being positive.
        - agg_labels: Dictionary. Aggregated labels. Each key is a timestep and
        each value is the true label at that timestep. No true aggregation is 
        performed since these should be the same for each repeated timestep, 
        this function just grabs the first label for each repeated timestep so 
        shapes match for metric calculation.
    """
    # initialize dictionaries for probabilities, labels, and classifications
    all_probabilities = {} # raw
    agg_probabilities = {} # aggregated
    agg_labels = {}
    classifications = {}
    """
    print("Debug Info:")
    print(f"Length of probabilities: {len(all_probabilities)}")
    print(f"Length of sequence_lengths: {len(sequence_lengths)}")
    print(f"Length of start_timesteps: {len(start_timesteps)}")
    """
    # accumulate predictions for each timestep
    # loop through our parameter lists simultaneously
    for i, (label, prob, sequence_len, start_time) in enumerate(zip(labels,
                                                                    probabilities, 
                                                                    sequence_lengths,
                                                                    start_timesteps)):
        min_len = min(sequence_len, prob.shape[0]) # make sure we don't go out of bounds
        print(f" Processing Window {i}: Start {start_time}, Seq Len {sequence_len}, Prob Shape {prob.shape}")
        for t in range(min_len): # for each timestep in the sequence
            timestep = start_time + t # define timestep relative to start time
            print(f"Timestep {timestep}: Label {label[t]}, Prob {prob[t]}")
            # if the timestep isn't already a dict key, initialize it
            if timestep not in all_probabilities:
                all_probabilities[timestep] = []
                # add the label for this timestep now- no true aggregation 
                # needed bc these should be the same for each repeated timestep.
                # we just need to do this to make the shapes the same
                #print(label[t].item())
                agg_labels[timestep] = label[t].item()
            # add the prediction to the dictionary
            all_probabilities[timestep].append(prob[t])
    # aggregate
    # for each key value pair in predictions dictionary
    for timestep, probs in all_probabilities.items():
        # get the average probability
        avg_prob = np.mean(probs)
        # add to list of aggregated probabilities
        agg_probabilities[timestep] = avg_prob
        if avg_prob >= threshold:
            classification = 1
        else:
            classification = 0
        classifications[timestep] = classification
    
    return classifications, agg_probabilities, agg_labels

def split_data(dfSamples):
    # maintain even distribution of subjects and runs in train and test
    # and account for proportion of MW occurances in each subject to ensure 
    # train and test have balanced distribution of MW occurances
    # so model generalizes well to new subjects and runs
    
    # calculate MW proportion per subject
    sub_mw_proportions = dfSamples.groupby(["Subject","run_num"])["is_MW"].mean().reset_index()
    # rename cols for clarity
    sub_mw_proportions.columns = ["Subject", "run_num", "mw_proportion"]
    
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
    train_data = pd.merge(dfSamples, train_pairs, on=["Subject", "run_num"])
    test_data = pd.merge(dfSamples, test_pairs, on=["Subject", "run_num"])
    
    # verify distribution looks good
    train_mw_p = train_data["is_MW"].mean()
    test_mw_p = test_data["is_MW"].mean()
    
    print("Mean of is_MW in train set: ", train_mw_p)
    print("Mean of is_MW in test set: ", test_mw_p)
    return train_data, test_data

def combine_records(df :pd.DataFrame) ->pd.DataFrame:
    # written by Omar Awajan, adapted by Alaina Birney
    # create mask where True means curr row is different from previous row
    # across all feature columns
    mask = ( 
        (df['LX'] != df['LX'].shift()) 
        & (df['LY'] != df['LY'].shift()) 
        & (df['LPupil_normalized'] != df['LPupil_normalized'].shift()) 
        & (df['RX'] != df['RX'].shift()) 
        & (df['RY'] != df['RY'].shift()) 
        & (df['RPupil_normalized'] != df['RPupil_normalized'].shift())
        & (df["Lblink"] != df["Lblink"].shift())
        & (df["Rblink"] != df["Rblink"].shift())
        )
    # Group duplicates 
    df['combined'] = mask.cumsum()
    # filter to get the first occurance of each combined group
    df_filtered = df.loc[df.groupby("combined").head(1).index]
    # drop combined column
    df_filtered = df_filtered.drop(columns=["combined"])
    return df_filtered

def add_padding(batch):
    # unpack inputs and labels
    (inputs, labels) = zip(*batch)
    
    # calculate lengths of unpadded sequences (for masking)
    seq_lens = torch.tensor([len(inp) for inp in inputs])
    # add padding to match sequence length, convert input to tensor first
    # pad sequence adds zero padding to match largest sequence in batch
    padded_inputs = pad_sequence([inp.clone().detach() if isinstance(inp, torch.Tensor) else torch.tensor(inp) for inp in inputs], batch_first=True)
    padded_labels = pad_sequence([lbl.clone().detach() if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in labels], batch_first=True)
    return padded_inputs, padded_labels, seq_lens

# define custom Dataset class
class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=2500, step_size=250):
        """
        Params:
            - data: The dataset 
            - sequence_length: The sequence length for the LSTM
            - step_size: The step size for the sliding window
            - scaler: scaler object to use. if none specified, standardScaler will be used
            - fit_scaler: whether or not to fit scaler (only fit to training data)
            - columns_to_scale: The columns to apply the scaler to 
        """
        self.data = data
        self.sequence_length = sequence_length
        self.step_size = step_size
            
        # separate features and labels
        # exclude non feature columns from features, but keep in dataset for logocv
        self.features = data.drop(labels=["is_MW", "page_num", "run_num","sample_id",
                                   "tSample", "Subject", "tSample_normalized",
                                   "mw_proportion", "mw_bin"], axis=1).values
        self.labels = data["is_MW"].values

    
    def __len__(self):
        # num valid sequences relative to sequence length and step size
        return (len(self.data) - self.sequence_length) // self.step_size +1
    
    def __getitem__(self, idx):
        # prevent the creation of sequences of length 0- truncate sequences when they are too long and add padding to match length
        # get start and end of sliding window
        start_idx = idx * self.step_size
        end_idx = start_idx + self.sequence_length
        
        # skip incomplete sequences that might arise at the end of the dataset
        """
        if end_idx > len(self.features):
            print(f"Skipping sequence at index {idx}: end_idx ({end_idx}) exceeds data length ({len(self.features)}).")
            return None
        """
        #print(f"Index {idx}: start={start_idx}, end={end_idx}, available={len(self.features)}")
        """
        if start_idx >= len(self.features) or end_idx > len(self.features):
            raise ValueError(f"Invalid sequence at index {idx}: start={start_idx}, end={end_idx}, length={len(self.features)}")
        """
        # get sequence of features
        x = self.features[start_idx:end_idx]
        # get corresponding labels (same length as sequence because many to many)
        y = self.labels[start_idx:end_idx]
        
        """
        # results in a bunch of sequences full of 0s after minibatch 501
        if len(x) < sequence_length:
            padding_length = self.sequence_length - len(x)
            x = np.pad(x, ((0, padding_length), (0,0)), "constant")
            y = np.pad(y,(0,padding_length), "constant")
        """
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# define LSTM class

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_p = .3):
        super(LSTMModel, self).__init__()
        # LSTM takes input tensor of shape (batch_size, sequence_length, input_size)
        # bidirectional lstm with batch as first dimension
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  bidirectional=True, dropout = dropout_p)
        #self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.fc = torch.nn.Linear(hidden_size*2, hidden_size*2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = torch.nn.Linear(hidden_size*2, hidden_size*2)
        self.relu2 = nn.ReLU()
        # fc should be hidden size *2 because bidirectional - map hidden state to output size
        self.fc3 = torch.nn.Linear(hidden_size*2, output_size)
    def forward(self,x, seq_lens):
        # x shape: (batch size, sequence length, number of features)
        # pack padded sequences - parameters (input, sequence lengths (must be on cpu), 
        # batch first to match input order, enforce sorted False because seqs aren't sorted
        # by length in decreasing order necessarily)
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lens.cpu(),
                                                               batch_first=True, enforce_sorted=False)
        #out, _ = self.lstm(x)
        # feed lstm packed input
        packed_out, _ = self.lstm(packed_x)
        # unpack output
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) 
        #out = self.layer_norm(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

#%% old- before splitting was done in separate script


file_path = "E:\\MW_Classifier_Data\\all_subjects_interpolated.csv"
dfSamples = load_data(file_path)


# check for missing values and confirm dtypes are correct
print(dfSamples.dtypes)
print(dfSamples.isna().sum())

# train test split
# do subject-wise train test split to ensure model generalizes well to new subjects
train_data, test_data = split_data(dfSamples)

# drop page num, run num, sample id, tSample, tSample_normalized, subject, mw_proportion from train and test data
train_data = train_data.drop(columns=["page_num", "run_num", "sample_id",
                                      "tSample", "Subject", "tSample_normalized",
                                      "mw_proportion", "mw_bin"])
test_data = test_data.drop(columns=["page_num", "run_num", "sample_id",
                                    "tSample", "Subject", "tSample_normalized",
                                    "mw_proportion", "mw_bin"])

print(train_data.columns)

#%% load and scale data

train_data_path = "C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\train.csv"
test_data_path = "C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\test.csv"

train_data = load_data(train_data_path)
test_data = load_data(test_data_path)

print(train_data.columns)
print(test_data.columns)

scaler = StandardScaler()
columns_to_scale = ["LX", "LY", "RX", "RY"]

train_scaled = train_data.copy()
test_scaled = test_data.copy()

# fit transform train
train_scaled.loc[:,columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
# transform test
test_scaled.loc[:,columns_to_scale] = scaler.transform(test_data[columns_to_scale])


#%%
# find mean mw duration in train set
# make copy of train data so i dont alter the actual data
train_copy = train_data.copy()
# find groups of consecutive rows where is_MW == 1
train_copy["MW_change"] = train_copy["is_MW"].ne(train_copy["is_MW"].shift()).cumsum()
# filter only mw = 1
mw_eps = train_copy[train_copy["is_MW"]==1]
# group by MW_change to id distinct episodes
mw_ep_lens = mw_eps.groupby("MW_change").size()
mean_mw_dur = mw_ep_lens.mean()
print(mean_mw_dur)

#%%
# seq len set very small for testing
# create datasets
# set parameters
sequence_length = 2500
# might have been suffering from vanishing gradient with 4k and 8k
step_size = 250

# instantiate 
train_dataset = WindowedTimeSeriesDataset(train_scaled, sequence_length,step_size)
test_dataset = WindowedTimeSeriesDataset(test_scaled, sequence_length,step_size)
# num training sequences
print(len(train_dataset))

#%% set up weightedRandomSampler for sequences


# calculate the majority label in each sequence
majority_labels = []
i = 1
print("calculating majority labels")
for idx, (_, sequence_labels) in enumerate(train_dataset):
   #if len(sequence_labels) == 0:
    #   break
    if idx >= len(train_dataset):
        break
    print(f"iteration {i} of {len(train_dataset)}")
    print(f"idx is {idx} and dataset length is {len(train_dataset)}")
    print(f"Start index: {idx * step_size}, End index: {idx * step_size + sequence_length}")
    print(f"Shape of sequence_labels: {sequence_labels.shape}")
    # using .5 threshold- if more than half the labels in the sequence are mw, label sequence as mw
    majority_label = (sequence_labels.sum() > (len(sequence_labels) //2)).int().item()
    majority_labels.append(majority_label)
    i+=1
    
majority_labels = torch.tensor(majority_labels)
# get class weights
# they are based on inverse frequency of current class distribution
# but to further compensate for our imbalance a multiplier is applied to 
# weight of mw classes
print("getting class weights")
not_mw = (majority_labels == 0).sum()
mw = (majority_labels == 1).sum()
#multiplier = 1.15
class_weights = [1.0/not_mw, (1.0/mw)]
print("assigning weights")
# assign each sample a weight based on class & class weights
sequence_weights = torch.tensor([class_weights[label] for label in majority_labels],
                              dtype=torch.float)

print("instantiating sampler")
# instantiate weightedRandomSampler
weighted_sampler = WeightedRandomSampler(weights=sequence_weights,
                                         num_samples = len(sequence_weights), replacement=True)
# len majority labels should match len train_dataset
if len(majority_labels) != len(train_dataset):
    print("Mismatch between size of majority labels and dataset")
    print("Majority labels length:", len(majority_labels))
    print("Train datset length:", len(train_dataset))
#%%
# create DataLoaders - no shuffling as this is time series data
# specify num workers?
# use collate fntn to add padding when sequences vary in length (we have more
# samples for some subjects than others) 
batch_size = 128
trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False,
                         sampler=weighted_sampler, collate_fn = add_padding)

testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, 
                        collate_fn = add_padding)

"""
for i, (inputs, labels) in enumerate(trainloader):
    print(f"Batch {i}, raw input shape: {[len(inp) for inp in inputs]}")
    print(f"Batch {i}, padded input shape: {inputs.shape}")
    if inputs.shape[1] == 0:  # Check if sequence length is 0
        print(f"Error: Sequence length is 0 in batch {i}")
"""

#%%
# get sense of how imbalanced classes are 

# this is without weighted random sampler
print("without weighted sampler")
# get mw in train
train_mw_count = train_data['is_MW'].sum()
train_non_mw_count = len(train_data) - train_mw_count

# get mw in test
test_mw_count = test_data['is_MW'].sum()
test_non_mw_count = len(test_data) - test_mw_count

# Print the class distributions
print(f"Training Set: Mind Wandering: {train_mw_count}, Not Mind Wandering: {train_non_mw_count}")
print(f"Training Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")

print(f"Test Set: Mind Wandering: {test_mw_count}, Not Mind Wandering: {test_non_mw_count}")
print(f"Test Set Ratio (MW/Non-MW): {test_mw_count / test_non_mw_count:.2f}")

# imbalance is similar in train and test, but heavy in both (~10% of each dataset is MW)
# use weighted loss function to penalize more for incorrect predictions on minority class (MW)
# training set ratio MW/not MW = 2838679/30675671 (pre subject stratification)
# post subject stratification Mind Wandering: 2830876, Not Mind Wandering: 29478085

print("with weighted sampler")
# with weighted random sampler
train_mw_count = 0
train_non_mw_count = 0
for i,(inputs,labels, seq_lens) in enumerate(trainloader):
    mw_count = (labels==1).sum().item()
    non_mw_count = (labels==0).sum().item()
    
    train_mw_count += mw_count
    train_non_mw_count += non_mw_count
    
print("train")
print("MW:", train_mw_count)
print("Non MW: ", train_non_mw_count)
print(f"Training Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")


# classes are still slightly imbalanced after using weighted random sampler
# maybe try including pos weight as well once best sequence length is found

#%%
# sliding window - applied during training data preparation
print(torch.__version__)            # Check PyTorch version
print(torch.cuda.is_available())    # Check if PyTorch detects CUDA
print(torch.cuda.device_count()) 
# check if GPU available and move model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

#%%
#del model
#del optimizer
#del loss


torch.cuda.empty_cache()

import gc
gc.collect()

# if memory issues persist after running, do nvidia-smi in terminal, kill
# the python PDI (Stop-Process -Id !PDI ID here! -Force), restart spyder, 
# then run this cell again


#%% Train
# set parameters
# TBTT - pyTorch does automatically when backprop over truncated sequences
#torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# set max grad norm for gradient clipping
max_grad_norm = 1.0

input_size = 6 # num features per timestep, num columns in train or test -1 for labels
hidden_size = 256
num_layers = 2 #2 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
num_epochs = 25
lr = .0005
dropout_p = .35
"""
# https://arxiv.org/pdf/1708.02002 focal loss paper

# get frequency of minority class
train_mw_count = 0
train_non_mw_count = 0
for i,(inputs,labels, seq_lens) in enumerate(trainloader):
    mw_count = (labels==1).sum().item()
    non_mw_count = (labels==0).sum().item()
    
    train_mw_count += mw_count
    train_non_mw_count += non_mw_count
    
mw_ratio = train_mw_count/ (train_non_mw_count + train_mw_count)
alpha = 1- mw_ratio  # the factor to increase weight of loss for positive class -use relative frequency of majority class

print("alpha: ", alpha)
if alpha > 1:
    print("WARNING: alpha is greater than 1.")
if alpha <0:
    print("WARNING: alpha is less than 0.")
gamma = 2 # the impact of focusing on hard examples (higher = more emphasis on hard) set to 1 since classes are almost balanced after WRS
"""
# following pos weight is for sequence len 4k batch size 64 step size 500- double check before using again though, might have been typo to have 1 on the end for not mw
#pos_weight = torch.tensor([294780851/2830876]).to(device) # pos weight is ratio of not MW/ MW to give more weight to pos class
# for seq len 8k, batch size 32, step size 250
#pos_weight = torch.tensor([29478085/2830876]).to(device) # pos weight is ratio of not MW/ MW to give more weight to pos class
#pos_weight = torch.tensor([31001855/2883141]).to(device) # pos weight is ratio of not MW/ MW to give more weight to pos class
#pos_weight = torch.tensor([180176844/158628156]).to(device)

# instantiate LSTM and move to GPU
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_p).to(device)

#criterion = nn.BCEWithLogitsLoss() # BCE with logits bc binary classification
criterion = nn.BCEWithLogitsLoss()
# pos weight to help with class imbalance
optimizer = optim.Adam(model.parameters(), lr=lr)

# train
# initialize list to store loss for each epoch
loss_vals = []

total_batches = len(trainloader)
# start timer
start_time = time.time()
for epoch in range(num_epochs):
    model.train() # put in training mode
    running_loss = 0.0
    epoch_start_time = time.time()
    for i, (inputs,labels, seq_lens) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        seq_lens = seq_lens.to(device)
        
        # zero gradients for this batch
        optimizer.zero_grad()
        # forward prop - dont need signmoid bc included in loss fntn
        outputs = model(inputs, seq_lens) # pass sequence lengths as well for packing padding
        # reshape labels to calculate loss
        loss = criterion(outputs, labels.unsqueeze(-1))
        #loss = sigmoid_focal_loss(outputs, labels.unsqueeze(-1), alpha, gamma, reduction="mean")
        #backprop
        loss.backward()
        # apply gradient clipping to LSTM layer only
        torch.nn.utils.clip_grad_norm(model.lstm.parameters(), max_grad_norm)
        optimizer.step()
        
        # accumulate loss
        running_loss += loss.item()
        
        # output stats
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            avg_batch_time = elapsed_time / ((epoch * total_batches) + (i+1))
            total_time = avg_batch_time * num_epochs * total_batches
            remaining_time = total_time - elapsed_time
            print("Epoch: %d Minibatch %5d loss: %.3f" %(epoch +1, i+1, loss.item()))
            print(f"Elapsed time: {elapsed_time //60:.0f} min {elapsed_time % 60:.0f} sec")
            print(f"Estimated remaining time: {remaining_time // 60:.0f} min {remaining_time % 60:.0f} sec")
            
    # get and store avg loss for this epoch
    epoch_loss = running_loss / len(trainloader)
    loss_vals.append(epoch_loss)
print("Training Complete")

# save the model
curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"C:\\Users\\abirn\\OneDrive\\Desktop\\Models\\MW_Classifier{curr_datetime}.pth"
torch.save(model.state_dict(), save_path)
print("Model saved to ", save_path)

# plot loss and save plot
plt.plot(range(1,num_epochs+1), loss_vals)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
# get datetime for saving fig

plt.savefig(f"C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\plots\\LSTM_loss_{curr_datetime}.png")
plt.show()

    
#%% Load saved model if needed

# replace end of path with datetime model was saved
model_path = "C:\\Users\\abirn\\OneDrive\\Desktop\\Models\\MW_Classifier2024-10-23_01-48-06.pth"
input_size = len(train_data.columns)-1 # num features per timestep, num columns in train or test -1 for labels
hidden_size = 256 
num_layers = 2 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(model_path))

#%% Evaluate 
# prior cells must be run so seq len and step size are defined
# load model if evaluating a saved model- code not in place yet
# evaluate on the test set
correct = 0
total = 0
model.eval() # put in evaluation mode

# initialize lists to hold raw predictions (not aggregated), probabilities, 
# labels, start timesteps for windows, and sequence lengths
raw_probabilities = []
all_labels = []
#all_probabilities = [] no longer needed, aggregating
sequence_lengths = []
start_timesteps = []

# initialize f1, auc, mcc for binary classification
f1 = F1Score(task="binary")
auc = AUROC(task="binary")
auc_pr = BinaryAUROC()
mcc = MatthewsCorrCoef(task="binary")

for window_idx, data in enumerate(testloader):
    with torch.no_grad(): # disable gradient calculation
        inputs, labels, seq_lens = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        seq_lens = seq_lens.to(device) # lengths of individual sequences in this batch
        
        # forward prop
        outputs = model(inputs, seq_lens)
        # remove extra dimension
        outputs = outputs.squeeze(-1)
        # get probabilities
        probabilities = torch.sigmoid(outputs).cpu()

        for i in range(len(seq_lens)):
            raw_probabilities.append(probabilities[i].numpy())
            all_labels.append(labels[i].cpu().numpy())
            sequence_lengths.append(seq_lens[i].item())
            start_timesteps.append((window_idx * batch_size + i)* step_size)

# max start_timesteps should be len(test data)-sequence_length (8515249)
# size of start_timesteps should be equal to our number of windows (34052)
"""
print(min(start_timesteps))
print(max(start_timesteps))
"""



# aggregate classifications across overlapping windows
#start_timesteps = [item for sublist in start_timesteps for item in sublist]
agg_classes, agg_probs, agg_labels = aggregate_probabilities(labels = all_labels,
                                                             probabilities = raw_probabilities,
                                                             sequence_lengths = sequence_lengths,
                                                             start_timesteps = start_timesteps)


# convert aggregated classifications and probabilities to tensors
shared_keys = sorted(agg_labels.keys())
aggregated_classifications = torch.tensor([agg_classes[t] for t in shared_keys])
aggregated_probabilities = torch.tensor([agg_probs[t] for t in shared_keys])
aggregated_labels = torch.tensor([agg_labels[t] for t in shared_keys])

"""
# print max key in agg_probs, labels, classes they should all be 8515249
max_prob_timestep = max(agg_probs.keys())
print(f"Max timestep in agg_probs: {max_prob_timestep}")
max_class_timestep = max(agg_classes.keys())
print(f"Max timestep in agg_classes: {max_class_timestep}")
max_label_timestep = max(agg_labels.keys())
print(f"Max timestep in agg_labels: {max_label_timestep}")
"""
# get auc
overall_auc = auc(aggregated_probabilities, aggregated_labels)
print(f"AUC: {overall_auc:.4f}")

# get auc-pr
# flatten probs and labels
flat_probs = aggregated_probabilities.view(-1)
flat_labs = aggregated_labels.view(-1)
auc_pr.update(flat_probs, flat_labs)
auc_pr_result = auc_pr.compute()
print(f"AUC-PR: {auc_pr_result:.4f}")
# get f1
overall_f1 = f1(aggregated_classifications, aggregated_labels)
print(f"Overall F1 Score: {overall_f1: .4f}")
# take accuracy with a grain of salt bc our classes are imbalanced
# accuracy = (correct/total)*100 pre-aggregation
accuracy = (aggregated_classifications == aggregated_labels).float().mean().item() * 100
print(f"Accuracy: {accuracy: .2f}%") 

# confusion matrix
# adapted from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
flat_classifications = aggregated_classifications.view(-1)
conf_mat = confusion_matrix(flat_labs, flat_classifications)
conf_df = pd.DataFrame(conf_mat / np.sum(conf_mat, axis=1)[:, None], index = ["Not MW", "MW"],
                     columns = ["Not MW", "MW"])

plt.figure(figsize=(15,10))
sn.heatmap(conf_df, annot=True, cmap="coolwarm", vmin = 0, vmax = 1)
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

plt.savefig(f"C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\plots\\LSTM_confmatrix_{curr_datetime}.png")

precision = precision_score(flat_labs, flat_classifications)
recall = recall_score(flat_labs, flat_classifications)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
# mcc
mcc_score = mcc(aggregated_classifications, aggregated_labels)
print("MCC", mcc_score)
# log loss
# make probabilities shape match labels shape
log_probs = aggregated_probabilities.squeeze(-1)
criterion = nn.BCELoss() # use BCELoss here because aggregated probabilities is 
# averaged probabilities for each timestep, no need for more sigmoid
log_loss = criterion(log_probs, aggregated_labels)
print(f"Log loss: {log_loss: .4f}")

print("Last 5 classifications and truth labels:")
for i in range(-5,0):
    # printing last item in sequence for five sequences of predictions and the corresponding label
    print(f"Classification: {int(aggregated_classifications[i].item())}, Truth: {int(aggregated_labels[i].item())}")

# plot truth labels vs classifications for each subject and run
# generate f1 for each subject and run