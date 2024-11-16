#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:22:29 2024

@author: Alaina
WILL NEED TO BASE VAL SPLIT ON OG_SUBJECT IF USING ONE HOT DATA
DOES NOT CONTAIN DATASET CLASS CHANGES FOR ONE HOT DATA
"""
# DONT FORGET TO CHANGE ALL "CV#" TO CV CORRESPONDING TO THIS RUN
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
import sys
import joblib
import csv
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
    
def save_loss(train_losses, val_losses, fold_number, filename="./Output/cv2/all_fold_losses.csv"):
    """
    
    Saves average training and validation losses for each epoch for each fold
    to a csv so that average train loss over epochs and average val loss over
    epochs can be plotted by averaging over folds.
    
    The csv is saved within the Output folder in the MW_Classifier folder.

    Parameters
    ----------
    train_losses : List of float. Length = num epochs.
        Average train loss for each epoch. 
    val_losses : List of float. Length = num epochs.
        Average vaidation loss for each epoch.
    fold_number : Int
        The fold number corresponding to the current training run. 

    Returns
    -------
    None.

    """
    # check if file exists, if it doesnt, create a new file
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file: # will create file if it doesnt exist
        writer = csv.writer(file)
        # write header if file didnt already exist
        if not file_exists:
            writer.writerow(["Fold", "Epoch", "Train_Loss", "Validation_Loss"])
        # write current fold losses
        for epoch in range(len(train_losses)):
            writer.writerow([fold_number+1, epoch+1, train_losses[epoch], val_losses[epoch]])
        
        # output
        print(f"Losses for fold {fold_number+1} saved to {filename}")
    
def process_one_fold(fold_number, logo, train_data, columns_to_scale, seq_len,
                     step_size, device, multi_gpu, dropout_percent):
    
    for fold, (train_idx, val_idx) in enumerate(logo.split(train_data, train_data["is_MW"],
                                                           groups)):
        # skip folds that don't correspond to specified fold number
        if fold != fold_number:
            continue 
        print("Processing fold ", fold+1, "...")
        print(f"Train fold size: {len(train_idx)}, Val fold size: {len(val_idx)}")
        # define total folds for time estimation
        total_folds = 15 # 19 subjects - 4 test subjects -> 15 folds
        
        # create train and validation sets for this fold
        train_fold = train_data.iloc[train_idx]
        val_fold = train_data.iloc[val_idx]
        
        # get subjects for this fold
        train_subjects = train_fold["Subject"].unique()
        val_subjects = val_fold["Subject"].unique()
        print(f"Train Subjects: {train_subjects}")
        print(f"Val Subject: {val_subjects}")
        
        train_fold_scaled = train_fold.copy()
        val_fold_scaled = val_fold.copy()
        
        # fit scaler on training data for this fold
        # initialize scaler
        scaler = StandardScaler()
        # fit transform train
        train_fold_scaled.loc[:,columns_to_scale] = scaler.fit_transform(train_fold[columns_to_scale])
        # transform validation
        val_fold_scaled.loc[:,columns_to_scale] = scaler.transform(val_fold[columns_to_scale])
        
        # prepare datasets for this fold
        train_fold_dataset = WindowedTimeSeriesDataset(train_fold_scaled, sequence_length, step_size)
        val_fold_dataset = WindowedTimeSeriesDataset(val_fold_scaled, sequence_length, step_size)
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
        # get class weights
        # they are based on inverse frequency of current class distribution
        # but to further compensate for our imbalance a multiplier is applied to 
        # weight of mw classes
        not_mw = (majority_labels == 0).sum()
        mw = (majority_labels == 1).sum()
        #multiplier = 1.15 removing multiplier bc not consistently needed across folds
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
        
        # get sense of how imbalanced classes are 

        print("Without weighted sampler...")
        # get mw in train
        train_mw_count = train_data['is_MW'].sum()
        train_non_mw_count = len(train_data) - train_mw_count

        # Print the class distributions
        print(f"Training Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")

        print("With weighted sampler...")
        # with weighted random sampler
        train_mw_count = 0
        train_non_mw_count = 0
        for i,(inputs,labels, seq_lens) in enumerate(trainloader):
            mw_count = (labels==1).sum().item()
            non_mw_count = (labels==0).sum().item()
            
            train_mw_count += mw_count
            train_non_mw_count += non_mw_count
            
        print(f"Training Set Ratio (MW/Non-MW): {train_mw_count / train_non_mw_count:.2f}")
        """
        # set alpha and gamma for focal loss
        mw_ratio = train_mw_count/ (train_non_mw_count + train_mw_count)
        alpha = 1- mw_ratio  # the factor to increase weight of loss for positive class -use class frequency

        print("alpha: ", alpha)
        if alpha > 1:
            print("WARNING: alpha is greater than 1.")
        if alpha <0:
            print("WARNING: alpha is less than 0.")
        gamma = 2 # the impact of focusing on hard examples (higher = more emphasis on hard) 

        """
        # initialize model, optimizer, loss fntn
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_percent).to(device)
        # use dataparallel if multi gpu
        if multi_gpu:
            model = torch.nn.DataParallel(model)
            
        criterion = nn.BCEWithLogitsLoss()
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
                #loss = sigmoid_focal_loss(outputs, labels.unsqueeze(-1), alpha, gamma, reduction="mean")
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
                # time will be correct after first pass because then it'll account
                # for validation too. first one won't account for validation 
                if i % 100 == 0:
                    elapsed_time = time.time() - start_time
                    avg_batch_time = elapsed_time / ((epoch * len(trainloader)) + (i+1))
                    remaining_batches_current_fold = (num_epochs * len(trainloader)) - ((epoch * len(trainloader)) + (i+1))
                    remaining_time_current_fold = remaining_batches_current_fold * avg_batch_time
                    
                    # estimate across all jobs
                    curr_job = int(os.getenv("SLURM_ARRAY_TASK_ID", default=0)) +1
                    total_jobs = total_folds
                    remaining_jobs = total_jobs - curr_job
                    remaining_time_all_jobs = (remaining_time_current_fold + elapsed_time) * remaining_jobs
                    
                    # print output
                    print(f"Epoch: {epoch+1} | Minibatch {i+1} | Loss: {loss.item():.4f}")
                    print(f"Elapsed time: {elapsed_time //60:.0f} min {elapsed_time % 60:.0f} sec")
                    print(f"Estimated remaining time for current fold: {remaining_time_current_fold // 60:.0f} min {remaining_time_current_fold % 60:.0f} sec")
                    print(f"Estimated remaining time for all jobs: {remaining_time_all_jobs // 60:.0f} min {remaining_time_all_jobs % 60:.0f} sec")
                    
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
                    #loss = sigmoid_focal_loss(outputs, labels.unsqueeze(-1), alpha, gamma, reduction="mean")
                    running_val_loss += loss.item()
            # get and store avg val loss for this epoch
            epoch_val_loss = running_val_loss / len(valloader)
            val_losses.append(epoch_val_loss)
            # print epoch stats
            print(f"Fold {fold + 1} | Epoch {epoch + 1}| Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # plot training and validation loss for this fold
        plt.figure(figsize=(10,6))
        plt.plot(range(1,num_epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1,num_epochs+1), val_losses, label="Validation Loss")
        plt.title(f"Training and Validation Loss Over Epochs For Fold {fold_number+1}")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.legend()
        # set caption
        plt.savefig(f"./Plots/cv2/LSTM_loss_fold_{fold_number+1}.png")
        plt.show()
        
        # save model after training complete
        if multi_gpu:
            best_model_state = model.module.state_dict()
        else:
            best_model_state = model.state_dict()

        # save model and scaler
        save_path = f"./Models/c21/LSTM_fold_{fold_number+1}.pth"
        torch.save(best_model_state, save_path)
        print("Model saved to ", save_path)
        save_path =  f"./Models/cv2/Scaler_fold_{fold_number+1}.pk1"
        joblib.dump(scaler, save_path)
        print("Scaler saved to ", save_path)
        # save loss to csv
        save_loss(train_losses, val_losses, fold_number)
        # empty cache before processing next fold
        torch.cuda.empty_cache()


print("Loading data...")
file_path = "./train.csv"
train_data = load_data(file_path)

logo = LeaveOneGroupOut()
groups = train_data["Subject"].values

# get fold number 
if len(sys.argv) < 2:
    raise ValueError("Fold number must be provided.")
fold_number = int(sys.argv[1])

print(f"Processing fold {fold_number+1}...")

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
lr = .0001
sequence_length = 2500 # trying smaller even though i think we need at least 4k to capture temporal context in LSTM memory
# might have been suffering from vanishing gradient with 4k and 8k
step_size = 250
batch_size = 128
columns_to_scale = ["LX", "LY", "RX", "RY"]
dropout_percent = .4


# Train
try:
    process_one_fold(fold_number, logo, train_data, columns_to_scale, sequence_length,
                         step_size, device, multi_gpu, dropout_percent)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("CUDA out of memory.")
    else:
        raise e