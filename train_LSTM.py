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
import time
import os



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
    def __init__(self, data, sequence_length=2500, step_size=250, scaler=None, fit_scaler=False, columns_to_scale=None):
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
        # exclude non feature columns from features, but keep in dataset for logocv
        self.features = data.drop(labels=["is_MW", "page_num", "run_num","sample_id",
                                   "tSample", "Subject", "tSample_normalized",
                                   "mw_proportion", "mw_bin"], axis=1).values
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
        out = self.fc(out)
        #print(f"output after fc shape: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


file_path = "./all_subjects_interpolated.csv"
if not os.path.exists(file_path):
    print(f"Error: Data file not found at {file_path}")
#dfSamples = load_data(file_path)
dfSamples=pd.read_csv(file_path)

# train test split
# do subject-wise train test split to ensure model generalizes well to new subjects
train_data, test_data = split_data(dfSamples)


logo = LeaveOneGroupOut()
groups = train_data["Subject"].values



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
num_epochs = 25 
lr = .001
sequence_length = 2500 # trying smaller even though i think we need at least 4k to capture temporal context in LSTM memory
# might have been suffering from vanishing gradient with 4k and 8k
step_size = 250
batch_size = 64
columns_to_scale = ["LX", "LY", "RX", "RY"]
# initialize scaler
scaler = StandardScaler()

# Cross Validation loop

# initalize lists to store loss over each fold
train_losses_folds = []
val_losses_folds = []

# initialize best val loss to be high so we can track the model that performs best and save it
best_val_loss = float("inf")
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(logo.split(train_data, train_data["is_MW"],
                                                       groups)):
    print("Processing fold ", fold+1, "...")
    print(f"Train fold size: {len(train_idx)}, Val fold size: {len(val_idx)}")
    # create train and validation sets for this fold
    train_fold = train_data.iloc[train_idx]
    val_fold = train_data.iloc[val_idx]
    
    # prepare datasets for this fold
    train_fold_dataset = WindowedTimeSeriesDataset(train_fold, sequence_length,
                                                  step_size, scaler=scaler,
                                                  fit_scaler=True,
                                                  columns_to_scale = columns_to_scale)
    val_fold_dataset = WindowedTimeSeriesDataset(val_fold, sequence_length,
                                                  step_size, scaler=scaler,
                                                  fit_scaler=False,
                                                  columns_to_scale = columns_to_scale)
    # set up weightedRandomSampler for this folds train set
    # calculate the majority label in each sequence
    majority_labels = []
    i = 1
    print("calculating majority labels")
    for idx, (_, sequence_labels) in enumerate(train_fold_dataset):
       #if len(sequence_labels) == 0:
        #   break
        if idx >= len(train_fold_dataset):
            break
        #print(f"iteration {i} of {len(train_fold_dataset)}")
        #print(f"idx is {idx} and dataset length is {len(train_fold_dataset)}")
        #print(f"Start index: {idx * step_size}, End index: {idx * step_size + sequence_length}")
        #print(f"Shape of sequence_labels: {sequence_labels.shape}")
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
    if len(majority_labels) != len(train_fold_dataset):
        print("Mismatch between size of majority labels and dataset")
        print("Majority labels length:", len(majority_labels))
        print("Train datset length:", len(train_fold_dataset))
        
    # set up dataloaders for this fold
    # not all fold sizes are divisible by batch size, so drop the last batch
    # if it cant the the right size
    trainloader = DataLoader(train_fold_dataset, batch_size = batch_size,
                             sampler = weighted_sampler, shuffle=False,
                             collate_fn=add_padding, drop_last=True)
    valloader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=add_padding)
    
    # calculate class imbalance to get pos weight
    train_mw_count = 0
    train_non_mw_count = 0
    for i,(inputs,labels, seq_lens) in enumerate(trainloader):
        mw_count = (labels==1).sum().item()
        non_mw_count = (labels==0).sum().item()
        
        train_mw_count += mw_count
        train_non_mw_count += non_mw_count
    # define pos weight as tensor that is ratio of not mw in train/ mw in train 
    pos_weight = torch.tensor([train_non_mw_count/train_mw_count]).to(device)
    
    # initialize model, optimizer, loss fntn
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # use dataparallel if multi gpu
    if multi_gpu:
        model = torch.nn.DataParallel(model)
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # initialize lists to track loss
    train_losses = []
    val_losses = []
    
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
            loss = criterion(outputs, labels.unsqueeze(-1))
            #backprop
            loss.backward()
            # apply gradient clipping to LSTM layer only
            torch.nn.utils.clip_grad_norm(model.lstm.parameters(), max_grad_norm)
            optimizer.step()
            
            # accumulate loss
            running_loss += loss.item()
            
            # output stats
            # time will be correct after first pass because then it'll account
            # for validation too. first one won't account for validation 
            # also, this is just the time remaining for ONE FOLD. 
            if i % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_batch_time = elapsed_time / ((epoch * len(trainloader)) + (i+1))
                remaining_batches = total_batches - ((epoch * len(trainloader)) + (i+1))
                remaining_time = remaining_batches * avg_batch_time
                print("Epoch: %d Minibatch %5d loss: %.3f" %(epoch +1, i+1, loss.item()))
                print(f"Elapsed time: {elapsed_time //60:.0f} min {elapsed_time % 60:.0f} sec")
                print(f"Estimated remaining time: {remaining_time // 60:.0f} min {remaining_time % 60:.0f} sec")
                
        # get and store avg loss for this epoch
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        # validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels, seq_lens) in enumerate(valloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                seq_lens = seq_lens.to(device)
                outputs = model(inputs, seq_lens)
                loss = criterion(outputs, labels.unsqueeze(-1))
                running_val_loss += loss.item()
        # get and store avg val loss for this epoch
        epoch_val_loss = running_val_loss / len(valloader)
        val_losses.append(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if multi_gpu:
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()
            print("New best val loss found!")
        # output stats
        print(f"Fold {fold + 1} | Epoch {epoch + 1}| Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    # add losses for this fold to lists
    train_losses_folds.append(train_losses)
    val_losses_folds.append(val_losses)

# average loss over all folds
avg_train_loss_per_epoch = np.mean(train_losses_folds, axis=0)
avg_val_loss_per_epoch = np.mean(val_losses_folds, axis=0)

# save the best model
if best_model_state is not None:
    curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"./Models/LSTM_{curr_datetime}.pth"
    torch.save(best_model_state, save_path)
    print("Model saved to ", save_path)
else:
    print("No best model found.")

# plot loss and save plot
plt.figure(figsize=(10,6))
plt.plot(range(1,num_epochs+1), avg_train_loss_per_epoch, label="Train Loss")
plt.plot(range(1,num_epochs+1), avg_val_loss_per_epoch, label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()
# get datetime for saving fig

plt.savefig(f"./Plots/LSTM_loss_{curr_datetime}.png")
plt.show()