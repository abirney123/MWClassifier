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

#%%
import torchmetrics
print(torchmetrics.__version__)
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
    def __init__(self, data, sequence_length=4000, step_size=1000, scaler=None, fit_scaler=False, columns_to_scale=None):
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
        self.columns_to_scale = columns_to_scale
        
        if self.columns_to_scale is not None:
            self.column_indices = [data.columns.get_loc(col) for col in columns_to_scale]
            
        # separate features and labels
        self.features = data.drop(columns=["is_MW"]).values
        self.labels = data["is_MW"].values
        

        
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
            
        if fit_scaler:
            self.features[:, self.column_indices] = self.scaler.fit_transform(self.features[:,self.column_indices])
        else:
            self.features[:,self.column_indices] = self.scaler.transform(self.features[:,self.column_indices])

    
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
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

#%%


file_path = "E:\\MW_Classifier_Data\\all_subjects_interpolated.csv"
dfSamples = load_data(file_path)

#%%
# check for missing values and confirm dtypes are correct
print(dfSamples.dtypes)
print(dfSamples.isna().sum())

#counts = dfSamples.groupby(["Subject", "run_num"]).size().reset_index(name='counts')

# Display the counts for each subject-run pair
#print(counts)
#rare_cases = counts[counts["counts"] < 2]
#print("Subject-Run pairs with fewer than 2 rows of data:")
#print(rare_cases)


#%%
"""
not using this rn, results in too few instances of mw
# combine duplicate rows
print("df samples length before combining duplicates")
print(len(dfSamples))
print("mw instances before combining duplicates")
samples_mw_count = dfSamples['is_MW'].sum()
print(samples_mw_count)

dfSamples = combine_records(dfSamples)
print("df samples length after combining duplicates")
print(len(dfSamples))
print("mw instances after combining duplicates")
samples_mw_count = dfSamples['is_MW'].sum()
print(samples_mw_count)
"""
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

#%%
# find mean mw duration in train set
# make copy of train data so i dont alter the actual dataq
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
# create datasets
# set parameters
sequence_length = 2500 # trying smaller even though i think we need at least 4k to capture temporal context in LSTM memory
# might have been suffering from vanishing gradient with 4k and 8k
step_size = 250
columns_to_scale = ["LX", "LY", "RX", "RY"]
# initialize scaler
scaler = StandardScaler()
# instantiate - scaling applied in dataset class
train_dataset = WindowedTimeSeriesDataset(train_data, sequence_length,
                                          step_size, scaler = scaler,
                                          fit_scaler = True,
                                          columns_to_scale = columns_to_scale)
test_dataset = WindowedTimeSeriesDataset(test_data, sequence_length,
                                         step_size,scaler = scaler,
                                         fit_scaler = False,
                                         columns_to_scale = columns_to_scale)
# num sequencecs for training
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
print("getting class weights")
not_mw = (majority_labels == 0).sum()
mw = (majority_labels == 1).sum()
class_weights = [1.0/not_mw, 1.0/mw]
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
# samples for some subjects than others) - removing bc no longer necessary i think collate_fn=add_padding
trainloader = DataLoader(train_dataset, batch_size = 128, shuffle=False,
                         sampler=weighted_sampler, collate_fn = add_padding)

testloader = DataLoader(test_dataset, batch_size = 128, shuffle=False, 
                        collate_fn = add_padding)

"""
for i, (inputs, labels) in enumerate(trainloader):
    print(f"Batch {i}, raw input shape: {[len(inp) for inp in inputs]}")
    print(f"Batch {i}, padded input shape: {inputs.shape}")
    if inputs.shape[1] == 0:  # Check if sequence length is 0
        print(f"Error: Sequence length is 0 in batch {i}")
"""

#%%
# get sense of how imbalanced classes are - change pos weight if necessary!

# this is without weighted random sampler
"""
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
"""
# imbalance is similar in train and test, but heavy in both (~10% of each dataset is MW)
# use weighted loss function to penalize more for incorrect predictions on minority class (MW)
# training set ratio MW/not MW = 2838679/30675671 (pre subject stratification)
# post subject stratification Mind Wandering: 2830876, Not Mind Wandering: 29478085

# with weighted random sampler
train_mw_count = 0
train_non_mw_count = 0
for i,(inputs,labels, seq_lens) in enumerate(trainloader):
    mw_count = (labels==1).sum().item()
    non_mw_count = (labels==0).sum().item()
    
    train_mw_count += mw_count
    train_non_mw_count += non_mw_count
    
    
print("MW:", train_mw_count)
print("Non MW: ", train_non_mw_count)

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

input_size = len(train_data.columns)-1 # num features per timestep, num columns in train or test -1 for labels
hidden_size = 128#256
num_layers = 1 #2 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
num_epochs = 25 
lr = .001
# following pos weight is for sequence len 4k batch size 64 step size 500- double check before using again though, might have been typo to have 1 on the end for not mw
#pos_weight = torch.tensor([294780851/2830876]).to(device) # pos weight is ratio of not MW/ MW to give more weight to pos class
# for seq len 8k, batch size 32, step size 250
#pos_weight = torch.tensor([29478085/2830876]).to(device) # pos weight is ratio of not MW/ MW to give more weight to pos class
#pos_weight = torch.tensor([31001855/2883141]).to(device) # pos weight is ratio of not MW/ MW to give more weight to pos class
pos_weight = torch.tensor([180176844/158628156]).to(device)

# instantiate LSTM and move to GPU
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # BCE with logits bc binary classification
#criterion = nn.BCEWithLogitsLoss()
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

# replace curr_datetime with datetime model was saved
model_path = "C:\\Users\\abirn\\OneDrive\\Desktop\\Models\\MW_Classifier{curr_datetime}.pth"
input_size = len(train_data.columns)-1 # num features per timestep, num columns in train or test -1 for labels
hidden_size = 128 #256 
num_layers = 1#2 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(model_path))

#%% Evaluate 

# load model if evaluating a saved model- code not in place yet
# evaluate on the test set
correct = 0
total = 0
model.eval() # put in evaluation mode

# initialize lists to hold all predictions probabilities and labels
all_predictions = []
all_labels = []
all_probabilities = []

# initialize f1, auc, mcc for binary classification
f1 = F1Score(task="binary")
auc = AUROC(task="binary")
auc_pr = BinaryAUROC()
mcc = MatthewsCorrCoef(task="binary")

for i, data in enumerate(testloader):
    with torch.no_grad(): # disable gradient calculation
        inputs, labels, seq_lens = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        seq_lens = seq_lens.to(device)
        
        # forward prop
        outputs = model(inputs, seq_lens)
        #y_probs = torch.sigmoid(outputs)
        # convert probs to predictions with .5 threshold
        y_preds = (outputs >= .5).float()
        # remove extra dimension from predictions
        y_preds = y_preds.squeeze(-1)
        
        # add predictions and labels to lists
        all_predictions.append(y_preds.cpu())
        all_labels.append(labels.cpu())
        all_probabilities.append(outputs.cpu())
        

        # flatten predictions and labels
        y_preds_flat = y_preds.view(-1)
        labels_flat = labels.view(-1)
        # update total and correct
        total += labels_flat.size(0)
        correct += (y_preds_flat == labels_flat).sum().item()


# concat all predictions and labels into single tensors
all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)
all_probabilities = torch.cat(all_probabilities)

# get auc
overall_auc = auc(all_probabilities, all_labels)
print(f"AUC: {overall_auc:.4f}")

# get auc-pr
# flatten probs and labels
flat_probs = all_probabilities.view(-1)
flat_labs = all_labels.view(-1)
auc_pr.update(flat_probs, flat_labs)
auc_pr_result = auc_pr.compute()
print(f"AUC-PR: {auc_pr_result:.4f}")
# get f1
overall_f1 = f1(all_predictions, all_labels)
print(f"Overall F1 Score: {overall_f1: .4f}")
# take accuracy with a grain of salt bc our classes are imbalanced
accuracy = (correct/total)*100
print(f"Accuracy: {accuracy: .2f}%") 

# confusion matrix
# adapted from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
flat_predictions = all_predictions.view(-1)
conf_mat = confusion_matrix(flat_labs, flat_predictions)
conf_df = pd.DataFrame(conf_mat / np.sum(conf_mat, axis=1)[:, None], index = ["Not MW", "MW"],
                     columns = ["Not MW", "MW"])

plt.figure(figsize=(15,10))
sn.heatmap(conf_df, annot=True, cmap="coolwarm", vmin = 0, vmax = 1)
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

plt.savefig(f"C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\plots\\LSTM_confmatrix_{curr_datetime}.png")

precision = precision_score(flat_labs, flat_predictions)
recall = recall_score(flat_labs, flat_predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
# mcc
mcc_score = mcc(all_predictions, all_labels)
print("MCC", mcc_score)
# log loss
# make probabilities shape match labels shape
log_probs = all_probabilities.squeeze(-1)
criterion = nn.BCEWithLogitsLoss()
log_loss = criterion(log_probs, all_labels)
print(f"Log loss: {log_loss: .4f}")

print("Last 5 predictions and truth labels:")
for i in range(-5,0):
    # printing last item in sequence for five sequences of predictions and the corresponding label
    print(f"Prediction: {int(all_predictions[i, -1].item())}, Truth: {int(all_labels[i, -1].item())}")


