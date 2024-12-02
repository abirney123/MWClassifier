# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:12:48 2024

@author: abirn

https://medium.com/@mijanr/different-ways-to-combine-cnn-and-lstm-networks-for-time-series-classification-tasks-b03fc37e91b6 
used as guide for combining architectures
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.ops import sigmoid_focal_loss
import time
import os
import joblib



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


def add_padding(batch):
    # unpack inputs and labels
    (inputs, labels, subjects, timestamps, runs) = zip(*batch)
    
    # calculate lengths of unpadded sequences (for masking)
    seq_lens = torch.tensor([len(inp) for inp in inputs])
    #print("sequence lens before padding")
    #print(seq_lens)
    # add padding to match sequence length, convert input to tensor first
    # pad sequence adds zero padding to match largest sequence in batch
    padded_inputs = pad_sequence([inp.clone().detach() if isinstance(inp, torch.Tensor) else torch.tensor(inp) for inp in inputs], batch_first=True)
    padded_labels = pad_sequence([lbl.clone().detach() if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in labels], batch_first=True)
    #print(f"padded input shape: {padded_inputs.shape}")
    padded_subjects = [sub for sub in subjects] # not actually applying padding
    padded_timestamps = pad_sequence([torch.tensor(ts) for ts in timestamps], batch_first=True)
    padded_runs = pad_sequence([torch.tensor(rn) for rn in runs], batch_first=True)
    return padded_inputs, padded_labels, padded_subjects, padded_timestamps, padded_runs, seq_lens


def group_sequences(dataset):
    # group sequence indices by their majority class
    # accepts dataset
    # returns grouped_indices, a dictionary with two keys (one for each class)
    # and values as lists of indices for the sequences whos majority class matches the key
    grouped_indices = {}
    for i in range(len(dataset)):
        _, labels, _, _,_ = dataset[i] # dataset returns features, labels, subject, timestep, run
        seq_label = int(labels.mean() > .5) # find the majority class for the sequence w .5 threshold
        # if this label isnt already a key in the dict, initialize the key with a list as value
        if seq_label not in grouped_indices:
            grouped_indices[seq_label] = [] 
        # add this index to grouped indices under the seq label key
        grouped_indices[seq_label].append(i)
    return grouped_indices

def stratify_batches(grouped_indices, batch_size):
    # separate the lists of sequence indices from the keys of grouped_indices
    mw_idxs = grouped_indices[1]
    not_mw_idxs = grouped_indices[0]
    
    # find which group is smaller to balance batches
    min_group_size = min(len(not_mw_idxs), len(mw_idxs))
    # get half batch size (this will be num samples per class per batch)
    half_batch = batch_size // 2
    
    # create batches
    batches = []
    for i in range(0, min_group_size, half_batch):
        # create this batch- half not mw sequences, half mw sequences
        batch = not_mw_idxs[i:i+half_batch] + mw_idxs[i:i+half_batch]
        # shuffle 
        random.shuffle(batch)
        batches.append(batch)
        
    # skipping leftover sequences- they will all be from the majority class
    return batches

class StratifiedBatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = batches
        
    def __iter__(self):
        for batch in self.batches:
            yield batch
            
    def __len__(self):
        return len(self.batches)

def add_padding(batch):
    # unpack inputs and labels
    (inputs, labels, subjects, timestamps, runs) = zip(*batch)
    
    # calculate lengths of unpadded sequences (for masking)
    seq_lens = torch.tensor([len(inp) for inp in inputs])
    #print("sequence lens before padding")
    #print(seq_lens)
    # add padding to match sequence length, convert input to tensor first
    # pad sequence adds zero padding to match largest sequence in batch
    padded_inputs = pad_sequence([inp.clone().detach() if isinstance(inp, torch.Tensor) else torch.tensor(inp) for inp in inputs], batch_first=True)
    padded_labels = pad_sequence([lbl.clone().detach() if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in labels], batch_first=True)
    #print(f"padded input shape: {padded_inputs.shape}")
    padded_subjects = [sub for sub in subjects] # not actually applying padding
    padded_timestamps = pad_sequence([torch.tensor(ts) for ts in timestamps], batch_first=True)
    padded_runs = pad_sequence([torch.tensor(rn) for rn in runs], batch_first=True)
    return padded_inputs, padded_labels, padded_subjects, padded_timestamps, padded_runs, seq_lens


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
                                   "tSample", "Subject", "mw_proportion", "mw_bin", "tSample_normalized"], axis=1).values
        #self.features = data.drop(labels=["is_MW", "page_num", "run_num","sample_id",
                                   #"tSample", "OG_subjects", "tSample_normalized",
                                   #"mw_proportion", "mw_bin"], axis=1).values
        self.labels = data["is_MW"].values
        # save subject column for error analysis (OG subjects if one hot data)
        #self.subjects = data["OG_subjects"].values
        self.subjects = data["Subject"].values
        # save subject level timestamps for error analysis
        self.timestamps = data["tSample_normalized"].values
        self.runs = data["run_num"].values
        self.pages = data["page_num"].values
        #self.subjects = data["Subject"].values
        self.valid_indices = self.compute_valid_indices()
        
    def compute_valid_indices(self):
        # find valid indices so that windows only contain one subject and run
        valid_indices = []
        num_samples = len(self.data)
        
        subjects = np.array(self.subjects)
        runs=np.array(self.runs)
        pages=np.array(self.pages)
        
        # find starting indices
        for start_idx in range(0, num_samples - self.sequence_length +1, self.step_size):
            end_idx = start_idx + self.sequence_length
            
            # check if window spans mult subs or runs
            if (np.all(subjects[start_idx:end_idx] == subjects[start_idx]) and 
                np.all(runs[start_idx:end_idx] == runs[start_idx]) and
                np.all(pages[start_idx:end_idx] == pages[start_idx])):
                    valid_indices.append(start_idx)
        print(f"Total valid windows: {len(valid_indices)}")
        return valid_indices

    
    def __len__(self):
        # num valid sequences relative to sequence length and step size
        return (len(self.valid_indices))
    
    def __getitem__(self, idx):
        # prevent the creation of sequences of length 0- truncate sequences when they are too long and add padding to match length
        # get start and end of sliding window
        start_idx = self.valid_indices[idx]
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
        subject = self.subjects[start_idx: end_idx]
        timestamp = self.timestamps[start_idx:end_idx]
        run = self.runs[start_idx:end_idx]
        
        """
        # results in a bunch of sequences full of 0s after minibatch 501
        if len(x) < sequence_length:
            padding_length = self.sequence_length - len(x)
            x = np.pad(x, ((0, padding_length), (0,0)), "constant")
            y = np.pad(y,(0,padding_length), "constant")
        """
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), subject, timestamp, run

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
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = torch.nn.Linear(hidden_size*2, hidden_size*2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        # fc should be hidden size *2 because bidirectional - map hidden state to output size
        self.fc3 = torch.nn.Linear(hidden_size*2, output_size)
    def forward(self,x, seq_lens):
        # x shape: (batch size, sequence length, number of features)
        # pack padded sequences - parameters (input, sequence lengths (must be on cpu), 
        # batch first to match input order, enforce sorted False because seqs aren't sorted
        # by length in decreasing order necessarily)
        #print(f"Input shape to LSTM before packing: {x.shape}")
        #print(f"sequence lengths: {seq_lens}")
        self.lstm.flatten_parameters() # resolve warning about module weights not part of single contiguous chunk of memory
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
        #out = self.layer_norm(out)
        out = self.fc(out)
        #print(f"output after fc shape: {out.shape}")
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
    
class CNNModel(nn.Module):
    def __init__(self, input_size=2500):
        super(CNNModel, self).__init__()

        # First 1D Convolution layer
        # 6 features, 6 in channels
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=6, padding=0, stride=6)
        #conv1_out_size = ((input_size - 6) // 6) + 1
        #self.feature_size = 32*conv1_out_size # adjust first num to match out channels if needed
        
        # Second 1D Convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=0, stride=2)
        #conv2_out_size = ((conv1_out_size - 5) // 2) + 1
        #self.feature_size = 64*conv2_out_size # match out channels
        #self.final_seq_len = conv2_out_size
        
        # Third 1D Convolutional layer
        #self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=2)
        #conv3_out_size = ((conv2_out_size - 3) // 2) + 1

        # Fully connected layers
        # commented out as this will be used as feature extractor
        #self.fc1 = nn.Linear(32 * conv1_out_size,512)
        #self.fc2 = nn.Linear(512, 512)
        #self.fc3 = nn.Linear(512, 512)
        #self.fc4 = nn.Linear(512, 128)
        #self.fc5 = nn.Linear(128, 1)  # Output layer

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        #out = self.conv3(out)
        #out = F.relu(out)
        
        #out = out.view(out.size(0), -1)  # Flatten the tensor for LSTM input
        
        # Forward pass through fully connected layers
        #out = self.fc1(out)
        #out = F.relu(out)
        #out = self.fc2(out)
        #out = F.relu(out)
        #out = self.fc3(out)
        #out = F.relu(out)
        #out = self.fc4(out)
        #out = F.relu(out)
        #out = self.fc5(out)

        return out
    
    def compute_seq_len(self, input_seq_len):
        seq_len = input_seq_len
        for layer in [self.conv1, self.conv2]:
            seq_len = (((seq_len - layer.kernel_size[0]) // layer.stride[0]) +1)
        return seq_len
    
    
def get_grad_norm(model, multi_gpu):
    total_norm = 0
    if multi_gpu:
        parameters = model.module.parameters()
    else:
        parameters = model.parameters()
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def add_autoclip_gradient_handler(model, clip_percentile, multi_gpu):
    grad_history = []
    
    def autoclip_gradient():
        # find gradient norm
        obs_grad_norm = get_grad_norm(model, multi_gpu)
        grad_history.append(obs_grad_norm)
        
        # find clip value based on historical gradients
        clip_value = np.percentile(grad_history, clip_percentile)
        
        # clip
        if multi_gpu:
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), clip_value)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
    return autoclip_gradient

class CNNLSTM(nn.Module):
    def __init__(self, cnn_input_size, lstm_input_size, hidden_size, num_layers,
                 output_size, dropout_p):
        super(CNNLSTM, self).__init__()
        self.cnn = CNNModel(input_size = cnn_input_size)
        self.lstm = LSTMModel(input_size = lstm_input_size, hidden_size = hidden_size,
                              num_layers = num_layers, output_size = output_size, 
                              dropout_p=dropout_p)
    def forward(self, x, seq_lens):
        # change shape to batch size, input size, seq len
        # should first be shape batch size, seq len, input size
        #print(f"X shape before permuting: {x.shape}")
        #print("should be batch size, seq len, input size")
        x = x.permute(0,2,1)
        #print(f"X shape after permuting (CNN input): {x.shape}")
        #print("should be batch size, input size, seq len")
        out = self.cnn(x)
        out = out.permute(0,2,1)
        #print(f"shape for LSTM input: {out.shape}")
        #print("should be batch size, seq len, input size")
        # adjust seq lens to match new lengths after cnn
        #new_seq_lens = ((seq_lens - 6) // 6) + 1 #(old seq len - kernel size) / stride +1
        new_seq_lens = torch.tensor([self.cnn.compute_seq_len(seq) for seq in seq_lens],
                                    device = seq_lens.device)
        out = self.lstm(out, new_seq_lens)
        
        return out
            
file_path = "./train_balanced_newstrat_10k.csv"
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

#max_grad_norm = 1.0

sequence_length = 2500
cnn_input_size = sequence_length
lstm_input_size=64 # out channels
# 32 out channels * ((input len - kernel size) / stride) +1
hidden_size = 256
num_layers = 2 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
num_epochs = 100
lr = .00005
 # trying smaller even though i think we need at least 4k to capture temporal context in LSTM memory
# might have been suffering from vanishing gradient with 4k and 8k
step_size = 250
batch_size = 64
columns_to_scale = ["LX", "LY", "RX", "RY", "LPupil_normalized", "RPupil_normalized"]# - for no sccades
#columns_to_scale = ["LX", "LY", "RX", "RY", "ampDeg_L", "ampDeg_R", "vPeak_L", "vPeak_R", "LPupil_normalized", "RPupil_normalized"]
dropout_percent = 0


# prepare data
train_scaled = train_data.copy()
train_scaled[columns_to_scale] = train_scaled[columns_to_scale].astype("float64")
scaler = StandardScaler()
# fit transform train
train_scaled.loc[:,columns_to_scale] = scaler.fit_transform(train_scaled[columns_to_scale])
"""
# scale tSample_norm- min max within run because trials are independent/ different lengths
# will need to be done separately on test set because those trials are also independent/ different lengths
time_scaler = MinMaxScaler()

train_scaled["tSample_normalized"] = train_scaled.groupby(["Subject", "run_num"])["tSample_normalized"].transform(
    lambda x: time_scaler.fit_transform(x.values.reshape(-1,1)).flatten())
"""
# verify that runs start at 0 and end at 1
"""
print("verifying scaling time worked correctly...")
print("each run should start at 0 and end at 1")
for subject in train_scaled["Subject"].unique():
    for run in train_scaled["run_num"].unique():
        run_data = train_scaled[train_scaled["run_num"] == run]
        print(f"Subject {subject}: Run {run}: Start = {run_data['tSample_normalized'].min()}, End = {run_data['tSample_normalized'].max()}")
"""

# prepare dataset
train_dataset = WindowedTimeSeriesDataset(train_scaled, sequence_length, step_size)
    
# get grouped indices
grouped_indices = group_sequences(train_dataset)
# create stratified batches
stratified_batches = stratify_batches(grouped_indices, batch_size)
print("Number of batches generated:")
print(len(stratified_batches))
   # set up dataloader
#trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False, collate_fn = add_padding)
# set up dataloader with stratified batching
trainloader = DataLoader(train_dataset, batch_sampler = StratifiedBatchSampler(stratified_batches), shuffle=False,collate_fn=add_padding)

# initialize model, optimizer, loss fntn
model = CNNLSTM( cnn_input_size, lstm_input_size, hidden_size, num_layers,
                output_size, dropout_percent).to(device)
# use dataparallel if multi gpu
if multi_gpu:
    model = torch.nn.DataParallel(model)

#pos_weight = torch.tensor([294780851/2830876]).to(device) 
# pos weight is ratio of not MW/ MW to give more weight to pos class - get from original df
mw_count = (train_data["is_MW"] == 1).sum()
not_mw_count = (train_data["is_MW"] == 0).sum()

if not_mw_count > mw_count:
    print("not mw is majority class")
    print(f"Pos weight value: not_mw_count/mw_count * 1.5: {not_mw_count/(mw_count * 1.5)}")
    pos_weight = torch.tensor([not_mw_count/(mw_count * 1.5)]).to(device)
elif mw_count > not_mw_count:
    print("mw is majority class")
    print(f"Pos weight value: mw_count/not_mw_count: {mw_count/not_mw_count}")
    pos_weight = torch.tensor([mw_count/not_mw_count]).to(device)
elif mw_count == not_mw_count:
    print("classes are balanced, using 1 as pos weight")
    pos_weight = torch.tensor([1]).to(device)


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#criterion = nn.BCEWithLogitsLoss()
if multi_gpu:
    optimizer = optim.AdamW(model.module.parameters(), lr=lr) #.01 weight decay is default
else:
    optimizer = optim.AdamW(model.parameters(), lr=lr) #.01 weight decay is default

# initialize list to track loss
train_losses = []

# initialize autograd clipping handler
clip_percentile = 95
autoclip_gradient = add_autoclip_gradient_handler(model, clip_percentile, multi_gpu)

# calculating class imbalance to set alpha
# calculate based on full set rather than batches
train_mw_count = (train_data["is_MW"] == 1).sum()
train_non_mw_count = (train_data["is_MW"] == 0).sum()

    
print(f"Training Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")
print(f"MW: {train_mw_count}")
print(f"Not MW: {train_non_mw_count}")
"""
p_mw = train_mw_count / (train_mw_count + train_non_mw_count)
majority_weight = 1-p_mw
# use values for alpha dynamically based on class imbalance- apply multipliers to decay this value for testing

alpha = majority_weight
gamma = 2

print(f"alpha: {alpha}")
print(f"gamma: {gamma}")
"""
#train
# start timer
start_time = time.time()
total_batches = len(trainloader)
for epoch in range(num_epochs):
    model.train() # put in training mode
    running_loss = 0.0
    epoch_start_time = time.time()
    # train loop
    for i, (inputs,labels, _, _, _, seq_lens) in enumerate(trainloader):
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
        # unsqueeze
        labels = labels.unsqueeze(-1) 
        # and truncate to match new sequence lengths (seq len - kernel size) / stride + 1
        # can directly truncate bc no pooling so "downsampling" comes directly from stride skipping
        max_new_seq_len = model.cnn.compute_seq_len(sequence_length)
        labels = labels[:,:max_new_seq_len]
        loss = criterion(outputs, labels)
        #loss = sigmoid_focal_loss(outputs, labels, alpha, gamma, reduction="mean")

        #backprop
        loss.backward()
        # apply autograd clipping
        autoclip_gradient()
        """
        apply gradient clipping to lstm layer only with set value for max grad norm 
        if multi_gpu:
            torch.nn.utils.clip_grad_norm_(model.module.lstm.parameters(), max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), max_grad_norm)
        """
        optimizer.step()
        
        # accumulate loss
        running_loss += loss.item()
        
        # output stats for minibatch
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
    train_losses.append(epoch_loss)

# plot training and validation loss for this fold
plt.figure(figsize=(10,6))
plt.plot(range(1,num_epochs+1), train_losses, label="Train Loss")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()

curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"./Plots/CNN-LSTM_loss_{curr_datetime}.png"
plt.savefig(save_path)
print("Loss plot saved to ", save_path)
plt.close()

# output stats
# save model and scaler
if multi_gpu:
    model_state = model.module.state_dict()
else:
    model_state = model.state_dict()
save_path = f"./Models/CNN-LSTM_{curr_datetime}.pth"
torch.save(model_state, save_path)
print("Model saved to ", save_path)
save_path =  f"./Models/CNN-LSTM_Scaler_{curr_datetime}.pk1"
joblib.dump(scaler, save_path)
print("Scaler saved to ", save_path)