# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:58:44 2024

@author: abirn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:53:46 2024

@author: Alaina
Data is in MW_Classifier folder

"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torchvision.ops import sigmoid_focal_loss
import time
import os
import joblib
import csv



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
    df_filtered = df_filtered.drop(labels=["combined"], axis=1)
    return df_filtered

def add_padding(batch):
    # unpack inputs and labels
    (inputs, labels) = zip(*batch)
    
    # calculate lengths of unpadded sequences (for masking)
    seq_lens = torch.tensor([len(inp) for inp in inputs])
    #print("sequence lens before padding")
    #print(seq_lens)
    # add padding to match sequence length, convert input to tensor first
    # pad sequence adds zero padding to match largest sequence in batch
    padded_inputs = pad_sequence([inp.clone().detach() if isinstance(inp, torch.Tensor) else torch.tensor(inp) for inp in inputs], batch_first=True)
    padded_labels = pad_sequence([lbl.clone().detach() if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in labels], batch_first=True)
    #print(f"padded input shape: {padded_inputs.shape}")
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
        self.layer_norm = nn.LayerNorm(hidden_size*2)
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
        #print(f"Input shape to LSTM before packing: {x.shape}")
        #print(f"sequence lengths: {seq_lens}")
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lens.cpu(),
                                                               batch_first=True, enforce_sorted=False)
        #print(f"Packed sequence data shape: {packed_x.data.shape}")
        #print(f"Packed sequence batch sizes: {packed_x.batch_sizes}")
        #out, _ = self.lstm(x)
        # feed lstm packed input
        packed_out, _ = self.lstm(packed_x)
        # unpack output
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) 
        #print(f"output after LSTM shape: {out.shape}")
        
        #print("Unpacked output shape ", out.shape)
        out = self.layer_norm(out)
        out = self.fc(out)
        #print(f"output after fc shape: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


file_path = "./train.csv"
if not os.path.exists(file_path):
    print(f"Error: Data file not found at {file_path}")

print("Loading data...")
train_data = load_data(file_path)

if "Unnamed: 0" in train_data.columns:
    train_data.drop("Unnamed: 0", axis=1, inplace=True)
    print("Unnamed: 0 dropped.") 
    print("train Columns: ", train_data.columns)
    

# Hyperparams & Initialize Scaler
# sliding window - applied during training data preparation
print(torch.__version__)            # Check PyTorch version
print(torch.cuda.is_available())    # Check if PyTorch detects CUDA
print(torch.cuda.device_count()) 
# check if GPU available and move model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device: ", device)
# check for multiple GPUs available
num_gpus = torch.cuda.device_count()
print(f"{num_gpus} gpus available")
# use dataparallel if mult gpus available
multi_gpu = num_gpus > 1

max_grad_norm = 1.0

# hard coding input size because cols are dropped in dataset instantiation so
# input size is no longer reflected by num columns
input_size = 6#len(train_data.columns)-1 # num features per timestep, num columns in train or test -1 for labels
hidden_size = 256
num_layers = 2 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
num_epochs = 15 
lr = .001
sequence_length = 2500 # trying smaller even though i think we need at least 4k to capture temporal context in LSTM memory
# might have been suffering from vanishing gradient with 4k and 8k
step_size = 250
batch_size = 128
columns_to_scale = ["LX", "LY", "RX", "RY"]
dropout_percent = .4


# prepare data
train_scaled = train_data.copy()
scaler = StandardScaler()
# fit transform train
train_scaled.loc[:,columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
    
# prepare dataset
train_dataset = WindowedTimeSeriesDataset(train_scaled, sequence_length, step_size)

# set up dataloader - removed weighted sampler
trainloader = DataLoader(train_dataset, batch_size = batch_size,
                         shuffle=False, collate_fn=add_padding, drop_last=True)

# calculating class imbalance to set alpha
train_mw_count = 0
train_non_mw_count = 0
for i,(inputs,labels, seq_lens) in enumerate(trainloader):
    mw_count = (labels==1).sum().item()
    non_mw_count = (labels==0).sum().item()
    
    train_mw_count += mw_count
    train_non_mw_count += non_mw_count
    
print(f"Training Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")

p_mw = train_mw_count / (train_mw_count + train_non_mw_count)
majority_weight = 1-p_mw
# use values for alpha dynamically based on class imbalance- apply multipliers to decay this value for testing

alpha_values = [majority_weight,.7*majority_weight, .3*majority_weight]
gamma_values = [1,1.5,2]
multipliers = [1,.7,.3]
print(alpha_values)

# create csv to store results
results_file = "./Output/focal_grid_search.csv"
with open(results_file, mode="w", newline="") as file:
    writer=csv.writer(file)
    writer.writerow(["Model", "Multiplier", "Alpha", "Gamma", "Epoch","Train_loss"])
model_count = 1 # to track how many models have been trained

# grid search loop

for idx, alpha in enumerate(alpha_values):
    multiplier = multipliers[idx]
    for gamma in gamma_values:
        print(f"Training with alpha= {alpha:.4f}, gamma= {gamma}")
        print(f"Multiplier = {multiplier}")
        print(f"Model {model_count} of {len(alpha_values)*len(gamma_values)}")
    
        # initialize model, optimizer, loss fntn
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_percent).to(device)
        # use dataparallel if multi gpu
        if multi_gpu:
            model = torch.nn.DataParallel(model)
            
        #criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # initialize lists to track loss
        train_losses = []
        
        # train and validate
        total_batches = len(trainloader) * num_epochs
        # start timer
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train() # put in training mode
            running_loss = 0.0
            epoch_start_time = time.time()
            # train loop
            for i, (inputs,labels, seq_lens) in enumerate(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                seq_lens = seq_lens.to(device)
                
                # zero gradients for this batch
                optimizer.zero_grad()
                # forward prop - dont need signmoid bc included in loss fntn
                #print(f"Inputs shape: {inputs.shape}, Sequence lengths: {seq_lens}")
                outputs = model(inputs, seq_lens) # pass sequence lengths as well for packing padding
                #print(f"Outputs shape: {outputs.shape}")
                # reshape labels to calculate loss
                #loss = criterion(outputs, labels.unsqueeze(-1))
                loss = sigmoid_focal_loss(outputs, labels.unsqueeze(-1), alpha, gamma, reduction="mean")
    
                #backprop
                loss.backward()
                # apply gradient clipping to LSTM layer only
                if multi_gpu:
                    torch.nn.utils.clip_grad_norm(model.module.lstm.parameters(), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm(model.lstm.parameters(), max_grad_norm)
                optimizer.step()
                
                # accumulate loss
                running_loss += loss.item()
                
                # output stats for minibatch
                if i % 100 == 0:
                    elapsed_time = time.time() - start_time
                    avg_batch_time = elapsed_time / ((epoch * len(trainloader)) + (i+1))
                    remaining_batches = total_batches - ((epoch * len(trainloader)) + (i+1))
                    remaining_time = remaining_batches * avg_batch_time
                    print("Epoch: %d Minibatch %5d loss: %.3f" %(epoch +1, i+1, loss.item()))
                    print(f"Elapsed time: {elapsed_time //60:.0f} min {elapsed_time % 60:.0f} sec")
                    print(f"Estimated remaining time for this model: {remaining_time // 60:.0f} min {remaining_time % 60:.0f} sec")
                    
            # get and store avg loss for this epoch
            epoch_loss = running_loss / len(trainloader)
            train_losses.append(epoch_loss)
            
            # log results
            with open(results_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([model_count, multiplier, alpha, gamma, epoch+1, epoch_loss])
    
        # plot training loss for this hyperparam combo
        plt.figure(figsize=(10,6))
        plt.plot(range(1,num_epochs+1), train_losses, label="Train Loss")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.legend()
    
        plt.savefig(f"./Plots/LSTM_loss_alpha_{alpha:.4f}_gamma{gamma}.png")
        plt.close()
    
        # output stats
        # save model and scaler
        if multi_gpu:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        save_path = f"./Models/LSTM_alpha_{alpha:.4f}_gamma_{gamma}.pth"
        torch.save(model_state, save_path)
        print("Model saved to ", save_path)
        save_path =  f"./Models/Scaler_alpha_{alpha:.4f}_gamma_{gamma}.pk1"
        joblib.dump(scaler, save_path)
        print("Scaler saved to ", save_path)
        model_count+=1 # increment counter
