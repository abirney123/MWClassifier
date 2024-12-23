# -*- coding: utf-8 -*-
"""
Created on Thur Dec 12 2024

@author: Omar A.Awajan
adapted from abriney
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
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

def load_data(file_path):
    # load data in chunks
    chunks = []
    chunk_count = 0
    total_rows = 0
    dfSamples = pd.read_csv(file_path)
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
                                   "tSample", "Subject", "tSample_normalized"], axis=1).values
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
 
    

class CONVLSTMModelwithAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_p=0.3, num_heads=4):
        super(CONVLSTMModelwithAttention, self).__init__()

        # Conv Layers
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=6, padding=0, stride=6)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2)

        # Dropout after Conv Layers
        self.conv_dropout = nn.Dropout(p=dropout_p)

        # Attention for Conv Layers
        self.conv_attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True)

        # LSTM Units
        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p
        )

        # Multi-Head Attention Mechanism
        self.num_heads = num_heads
        self.lstm_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, output_size)

        # Dropout before Fully Connected Layers
        self.fc_dropout = nn.Dropout(p=dropout_p)

        # Activation
        self.Relu = nn.ReLU()
        self.Gelu = nn.GELU()

    def forward(self, x, seq_lens):
        # Prepare input for Conv layers
        x = x.permute(0, 2, 1)  # [batch_size, num_channels, seq_length]

        # 1D-Conv layers
        out = self.conv1(x)
        out = self.Relu(out)
        out = self.conv2(out)
        out = self.Relu(out)

        # Apply dropout after Conv layers
        out = self.conv_dropout(out)

        # Apply attention to Conv layer output
        out = out.permute(0, 2, 1)  # [batch_size, seq_length, features]
        conv_attention_out, _ = self.conv_attention(out, out, out)  # [batch_size, seq_length, features]

        # LSTM
        lstm_out, _ = self.lstm(conv_attention_out)  # [batch_size, seq_length, hidden_size * 2]

        # Adding Residual Connection
        #lstm_out += conv_attention_out

        # Multi-Head Attention for LSTM output
        attention_out, _ = self.lstm_attention(lstm_out, lstm_out, lstm_out)  # [batch_size, seq_length, hidden_size * 2]

        # Pooling (weighted sum)
        context_vector = attention_out.mean(dim=1)  # [batch_size, hidden_size * 2]

        # Fully connected layers
        out = self.fc1(context_vector)
        out = self.Gelu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        out = self.Gelu(out)
        out = self.fc_dropout(out)  # Apply dropout before second FC layer
        out = self.fc3(out)

        return out

def main():

    # Define file path
    file_path = "D:\\train_balanced_newstrat_10k.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Data file not found at {file_path}")

    print("Loading data...")
    train_data = load_data(file_path)

    # Drop unnecessary columns
    if "Unnamed: 0" in train_data.columns:
        train_data.drop("Unnamed: 0", axis=1, inplace=True)
        print("Unnamed: 0 dropped.")
        print("Train columns:", train_data.columns)

    # Hyperparameters
    sequence_length = 2500
    cnn_input_size = sequence_length
    lstm_input_size = 64  # Output channels of Conv1D layers
    hidden_size = 256
    num_layers = 2
    output_size = 1
    num_epochs = 60
    lr = 0.00005
    step_size = 250
    batch_size = 32
    columns_to_scale = ["LX", "LY", "RX", "RY", "LPupil_normalized", "RPupil_normalized"]
    dropout_percent = 0.3

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} GPUs available.")
    multi_gpu = num_gpus > 1

    # Scale data
    train_scaled = train_data.copy()
    train_scaled[columns_to_scale] = train_scaled[columns_to_scale].astype("float64")
    scaler = StandardScaler()
    train_scaled[columns_to_scale] = scaler.fit_transform(train_scaled[columns_to_scale])

    # Prepare dataset and DataLoader
    train_dataset = WindowedTimeSeriesDataset(train_scaled, sequence_length, step_size)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=add_padding)

    # Initialize model
    model = CONVLSTMModelwithAttention(cnn_input_size, hidden_size, num_layers, output_size, dropout_percent).to(device)
    if multi_gpu:
        model = torch.nn.DataParallel(model)

    # Class imbalance handling
    mw_count = (train_data["is_MW"] == 1).sum()
    not_mw_count = (train_data["is_MW"] == 0).sum()
    if not_mw_count > mw_count:
        pos_weight = torch.tensor([not_mw_count / mw_count]).to(device)
    elif mw_count > not_mw_count:
        pos_weight = torch.tensor([mw_count / not_mw_count]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    start_time = time.time()
    total_batches = len(trainloader)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (inputs, labels, _, _, _, seq_lens) in enumerate(trainloader):
            inputs, labels, seq_lens = inputs.to(device), labels.to(device), seq_lens.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, seq_lens)
            labels = labels.mean(dim=1) # [batch_size, 1]
            labels = labels.unsqueeze(-1)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            if i % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_batch_time = elapsed_time / ((epoch * total_batches) + (i + 1))
                remaining_time = avg_batch_time * (num_epochs * total_batches - ((epoch * total_batches) + (i + 1)))
                print(f"Epoch {epoch + 1}, Batch {i + 1}/{total_batches}, Loss: {loss.item():.4f}, Remaining Time: {remaining_time // 60:.0f}m {remaining_time % 60:.0f}s")

        epoch_loss = running_loss / total_batches
        train_losses.append(epoch_loss)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    loss_plot_path = f"D:\\Plots\\CONVLSTM_Loss_{curr_datetime}.png"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

    # Save model and scaler
    model_save_path = f"D:\\Models\\CONVLSTM_Model_{curr_datetime}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    scaler_save_path = f"D:\\Models\\CONVLSTM_Scaler_{curr_datetime}.pkl"
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")



if __name__ == '__main__':
    main()