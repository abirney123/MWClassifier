# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:29:47 2024

@author: abirn
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchmetrics import F1Score, AUROC, MatthewsCorrCoef
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sn
import joblib
import torch.nn.functional as F


# hyperparams
# one hot 31 saccades 12 interp 6 + 1 to all when time is added as feature
input_size = 6
hidden_size = 256 
num_layers = 2 # more than 3 isn't usually valuable
output_size = 1 # how many values to predict for each timestep
dropout_percent = 0
batch_size = 128
step_size = 250
sequence_length = 2500
cnn_input_size = sequence_length
#lstm_input_size = 32 * (((cnn_input_size - 6) //6) +1) # num features per timestep
lstm_input_size=64 # match output size for cnn

# functions
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_p = dropout_percent):
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
        #self.dropout2 = nn.Dropout(dropout_p)
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
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        #out = self.dropout2(out)
        out = self.fc3(out)
        return out
    
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

# define custom Dataset class
class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=2500, step_size=250):
        """
        Params:
            - data: The dataset 
            - sequence_length: The sequence length for the LSTM
            - step_size: The step size for the sliding window
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


class WindowedTimeSeriesDatasetOLD(Dataset):
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
                                   "tSample", "Subject", "mw_proportion", "mw_bin"], axis=1).values
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
        #self.subjects = data["Subject"].values

    
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
    
    
def aggregate_probabilities(labels, probabilities, sequence_lengths, all_subjects, all_runs, all_timesteps, threshold=0.5):
    """
    Aggregate probabilities from multiple windows that cover the same
    timesteps to provide more robust classifications. Additionally, aggregate
    labels so they are comparable.
    Does this on an individual subject basis so that labels don't conflict
    """

    # flatten lists so each entry corresponds to one timestep to match all_timesteps
    # all arrays are len 93102500
    print("flattening")
    flat_labels = np.array(labels)
    flat_labels = flat_labels.flatten()
    #print("flat_labels shape")
    #print(flat_labels.shape)
    flat_probs = np.array(probabilities)
    flat_probs = flat_probs.flatten()
    #print("flat_probs shape")
    #print(flat_probs.shape)
    flat_subs = np.array(all_subjects)
    flat_subs = flat_subs.flatten()
    #print("flat_subs shape")
    #print(flat_subs.shape)
    flat_runs = np.array(all_runs)
    flat_runs = flat_runs.flatten()
    #print("flat_runs shape")
    #print(flat_runs.shape)
    flat_timesteps = np.array(all_timesteps)
    flat_timesteps = flat_timesteps.flatten()
    #print("flat_timesteps shape")
    #print(flat_timesteps.shape)
    
    print("Lengths of input arrays for DataFrame:")
    print(f"Subjects: {len(flat_subs)}")
    print(f"Runs: {len(flat_runs)}")
    print(f"Timestamps: {len(flat_timesteps)}")
    print(f"Labels: {len(flat_labels)}")
    print(f"Probabilities: {len(flat_probs)}")
    
    # create df
    df = pd.DataFrame({
    "Subject": flat_subs,
    "Run": flat_runs,
    "Timestep": flat_timesteps,
    "Label": flat_labels,
    "Probability": flat_probs
})

    print("averaging and converting")
    # group by sub, run, timestep to perform actions across pairings
    grouped = df.groupby(["Subject", "Run", "Timestep"])
    """
    print("check if probabilities are actually being averaged")
    unique_counts = grouped["Probability"].nunique()
    mult_probs = unique_counts[unique_counts > 1]
    timesteps_with_multiple_probs = mult_probs.groupby(["Subject", "Run"]).size()
    print("timesteps with mult probabilities for each subject run pair")
    print(timesteps_with_multiple_probs)
    
    Subject  Run
    10089    1.0    485750
             2.0    463000
             3.0    565963
             4.0    505056
             5.0    543804
    10110    1.0    585000
             2.0    488954
             3.0    449000
             4.0    523937
             5.0    511842
    10111    1.0    321000
             2.0    371000
             3.0    215000
             4.0    285000
             5.0    341000
    10117    1.0    547000
             2.0    582999
             3.0    492997
             4.0    543000
             5.0    490529
    """
    # get first label for each subject, run, timestep pairing- confirmed there are no conflicts
    agg_labels = grouped["Label"].first().to_dict()
    # do the same for the subject for each pairing
    agg_subs = grouped["Subject"].first().to_dict()
    # avg probs for each subject, run, timestep pairing
    agg_probs = grouped["Probability"].mean()
    # convert to classifications
    agg_classes = (agg_probs >= threshold).astype(int)
    # convert return vals to dictionaries
    
    classifications = agg_classes.to_dict()
    agg_probabilities = agg_probs.to_dict()
    
    """
    # Detect conflicts
    print("detecting conflicts")
    grouped = df.groupby(["Subject", "Run", "Timestep"])
    for group_name, group_data in list(grouped)[:5]:  # Limit to first 5 groups
        print(f"Group: {group_name}")
        print(group_data.head())
    
    agg_labels = {}
    for key, group in grouped:
        unique_labels = group["Label"].unique()
        if len(unique_labels) > 1:
            # conflict detected: multiple labels for the same key
            print(f"Conflict detected at {key}: Labels {unique_labels}")
        agg_labels[key] = unique_labels[0]  # Take the first label - if no conflicts this is same for each instance
    """
    
    return classifications, agg_probabilities, agg_labels, agg_subs

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

        

def plot_err_over_time(results, subject):
    sub_df = results[results["Subject"] == subject]
    unique_runs = sub_df["Run"].unique()
    num_runs = len(unique_runs)
    
    fig, axes = plt.subplots(num_runs, 1, figsize=(15,8), sharex=True)
    
    
    if num_runs == 1:
        print(f"Only one run for subject {subject}")
        axes=[axes]
        
    for i, run in enumerate(unique_runs):
        run_df = sub_df[sub_df["Run"] == run]
        
        # get error indices
        fp_idxs = run_df[run_df["Error_Type"] == "FP"]["Timestep"]
        fn_idxs = run_df[run_df["Error_Type"] == "FN"]["Timestep"]
        # plot errors
        axes[i].scatter(fp_idxs, [1]*len(fp_idxs), color="red", marker="+", label="False Positive")
        axes[i].scatter(fn_idxs, [0]*len(fn_idxs), color="blue", marker="_", label="False Negative")
        
        # subplot info
        axes[i].set_title(f"Run {run}")
        axes[i].set_yticks([0,1])
        axes[i].set_yticklabels(["False Negative", "False Positive"])
    # plot info
    plt.xlabel("Timestep Relative to Subject")
    fig.suptitle(f"Errors Over Time: Subject {subject}")
    plt.tight_layout(rect=[0,0,1,.9])
    #plt.legend(loc="upper right")
    plt.savefig(f"./Plots/Errors_over_time_s{subject}_CNNLSTM_12-2_2.png")
    plt.close()


# load data and scaler

#file_path = "E:\\MW_Classifier_Data\\test.csv"
file_path = "./test.csv"
test_data = load_data(file_path)

# initialize scaler

scaler = joblib.load("./Models/CNN-LSTM_Scaler_2024-12-02_11-07-50.pk1")
# Load saved model 
print(torch.__version__)            # Check PyTorch version
print(torch.cuda.is_available())    # Check if PyTorch detects CUDA
print(torch.cuda.device_count()) 
# check if GPU available and move model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# replace cv0 in path with str corresponding to cv run
model_path = "./Models/CNN-LSTM_2024-12-02_11-07-50.pth"
model = CNNLSTM( cnn_input_size, lstm_input_size, hidden_size, num_layers,
                output_size, dropout_percent).to(device)
model.load_state_dict(torch.load(model_path))
print("model loaded")
# apply scaler, define dataset and dataloader
columns_to_scale = ["LX", "LY", "RX", "RY", "LPupil_normalized", "RPupil_normalized"]
#columns_to_scale = ["LX", "LY", "RX", "RY", "ampDeg_L", "ampDeg_R", "vPeak_L", "vPeak_R"]
test_scaled = test_data.copy()
test_scaled[columns_to_scale] = test_scaled[columns_to_scale].astype("float64")
test_scaled.loc[:,columns_to_scale] = scaler.transform(test_scaled[columns_to_scale])
"""
time_scaler = MinMaxScaler()

test_scaled["tSample_normalized"] = test_scaled.groupby(["Subject", "run_num"])["tSample_normalized"].transform(
    lambda x: time_scaler.fit_transform(x.values.reshape(-1,1)).flatten())
"""
print("data scaled")
# get unique subjects for analysis later - use subject if no one hot, OG subject otherwise
test_subjects = test_data["Subject"].unique()
#test_subjects = test_data["OG_subjects"].unique()

test_dataset = WindowedTimeSeriesDataset(test_scaled, sequence_length, step_size)

testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, 
                        collate_fn = add_padding)

print("dataset and dataloader created, evaluating...")
# Evaluate 
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
all_subjects = []
all_timestamps = []
#all_probabilities = [] no longer needed, aggregating
sequence_lengths = []
start_timesteps = []
all_runs = []

# initialize f1, auc, mcc for binary classification
f1 = F1Score(task="binary")
auc = AUROC(task="binary")
mcc = MatthewsCorrCoef(task="binary")

for window_idx, data in enumerate(testloader):
    with torch.no_grad(): # disable gradient calculation
        inputs, labels, subjects, timestamps, runs, seq_lens = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        seq_lens = seq_lens.to(device) # lengths of individual sequences in this batch
        
        # forward prop
        outputs = model(inputs, seq_lens)
        # remove extra dimension
        outputs = outputs.squeeze(-1)
        # truncate to match shapes after cnn
        max_new_seq_len = outputs.size(1)
        labels = labels[:,:max_new_seq_len]
        # get probabilities
        probabilities = torch.sigmoid(outputs).cpu()
        labels = labels.cpu()

        for i in range(len(seq_lens)):
            # match lengths of appended items to truncated length from cnn
            truncated_len = probabilities[i].shape[0]
            raw_probabilities.append(probabilities[i].numpy())
            all_labels.append(labels[i, :truncated_len].cpu().numpy())
            all_subjects.append(subjects[i][:truncated_len])
            all_runs.append(runs[i][:truncated_len])
            all_timestamps.extend(timestamps[i][:truncated_len].cpu().numpy())
            sequence_lengths.append(seq_lens[i].item())
            #start_timesteps.append((window_idx * batch_size + i)* step_size)
        """
        # accumulate batch level data
        raw_probabilities.append(probabilities)
        all_labels.append(labels)
        all_subjects.extend(subjects)
        all_timestamps.extend(timestamps)
        all_runs.extend(runs)
        sequence_lengths.extend(seq_lens.tolist())
        """
        


print("aggregating")
# aggregate classifications across overlapping windows
#start_timesteps = [item for sublist in start_timesteps for item in sublist]
agg_classes, agg_probs, agg_labels, agg_subs = aggregate_probabilities(labels = all_labels,
                                                             probabilities = raw_probabilities,
                                                             sequence_lengths = sequence_lengths,
                                                             all_subjects = all_subjects,
                                                             all_runs = all_runs,
                                                             all_timesteps = all_timestamps)


# convert aggregated classifications and probabilities to tensors
shared_keys = sorted(agg_labels.keys())
aggregated_classifications = torch.tensor([agg_classes[key] for key in shared_keys])
aggregated_probabilities = torch.tensor([agg_probs[key] for key in shared_keys])
aggregated_labels = torch.tensor([agg_labels[key] for key in shared_keys])


# check that subjects and shared keys are the same size now
if len(agg_subs) != len(shared_keys):
    print(f"Size mismatch: Aggregated subjects length: {len(agg_subs)}, Shared Keys length: {len(shared_keys)}")

# and create df for subject and time analysis
results = pd.DataFrame({
    "Timestep": [key[2] for key in shared_keys],
    "True_Label": [agg_labels[key] for key in shared_keys],
    "Pred_Label": [agg_classes[key] for key in shared_keys],
    "Probability": [agg_probs[key] for key in shared_keys],
    "Run": [key[1] for key in shared_keys],
    "Subject": [agg_subs[key] for key in shared_keys]})

# add error type column - TP, FP, TN, or FN
results["Error_Type"] = "TP" # tp default
#results["Error_Type"] = results["Error_Type"].astype(object) # true pos default
results.loc[(results["True_Label"] == 1) & (results["Pred_Label"] == 1), "Error_Type"] = "TP"
results.loc[(results["True_Label"] == 1) & (results["Pred_Label"] == 0), "Error_Type"] = "FN"
results.loc[(results["True_Label"] == 0) & (results["Pred_Label"] == 1), "Error_Type"] = "FP"
results.loc[(results["True_Label"] == 0) & (results["Pred_Label"] == 0), "Error_Type"] = "TN"

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
# replace fold number with fold number for model being evaluated/ best model
plt.title("Confusion Matrix")

plt.savefig(f"./Plots/LSTM_confmatrix_CNNLSTM_12-2_2.png")

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

#test_subjects = [10089, 10110, 10111, 10117]
# analyze errors over time for each subject
for subject in test_subjects:
    plot_err_over_time(results, subject)