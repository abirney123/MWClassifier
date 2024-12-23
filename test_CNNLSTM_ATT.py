# -*- coding: utf-8 -*-
"""
Created on Thur Dec 12 2024

@author: Omar A.Awajan
adapted from abriney
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
import seaborn as sns
import joblib
import torch.nn.functional as F


# hyperparams
# one hot 31 saccades 12 interp 6 + 1 to all when time is added as feature
input_size = 6
hidden_size = 256 
num_layers = 2 # more than 3 isn't usually valuable
output_size = 1 # how many values to predict for each timestep
dropout_percent = 0
batch_size = 64
step_size = 250
sequence_length = 2500
cnn_input_size = sequence_length
lstm_input_size=64 # match output size for cnn

# functions

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

def plot_err_over_time(results, subject):
    sub_df = results[results["Subject"] == subject]
    unique_runs = sub_df["Run"].unique()
    num_runs = 5 #len(unique_runs)
    
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
    plt.savefig(f"D:\\Plots\\Errors_over_time_s{subject}_CNNLSTM_12-5_2.png")
    plt.close()

    
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
    # load data and scaler
    file_path = "D:\\test.csv"
    test_data = load_data(file_path)

    # initialize scaler
    scaler = joblib.load("D:\\Models\\CONVLSTM_Scaler_2024-12-13_08-05-29.pkl")

    # Check if GPU available
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Load saved model
    model_path = "D:\\Models\\CONVLSTM_Model_2024-12-13_08-05-29.pth"
    model = CONVLSTMModelwithAttention(2500, 256, 2, 1, 0.3).to(device)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded")

    # Apply scaler, define dataset and DataLoader
    columns_to_scale = ["LX", "LY", "RX", "RY", "LPupil_normalized", "RPupil_normalized"]
    test_scaled = test_data.copy()
    test_scaled[columns_to_scale] = test_scaled[columns_to_scale].astype("float64")
    test_scaled.loc[:, columns_to_scale] = scaler.transform(test_scaled[columns_to_scale])

    print("Data scaled")
    test_subjects = test_data["Subject"].unique()

    test_dataset = WindowedTimeSeriesDataset(test_scaled, sequence_length=2500, step_size=250)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=add_padding)

    print("Dataset and DataLoader created, evaluating...")

    # Evaluate
    model.eval()
    raw_probabilities = []
    all_labels = []
    all_subjects = []
    all_timestamps = []
    sequence_lengths = []
    all_runs = []

    f1 = F1Score(task="binary")
    auc = AUROC(task="binary")
    mcc = MatthewsCorrCoef(task="binary")

    for window_idx, data in enumerate(testloader):
        with torch.no_grad():
            inputs, labels, subjects, timestamps, runs, seq_lens = data
            inputs, labels, seq_lens = inputs.to(device), labels.to(device), seq_lens.to(device)

            # Forward pass
            outputs = model(inputs, seq_lens).squeeze(-1)
            labels = labels.mean(dim=1).unsqueeze(-1)  # Adjust target shape

            probabilities = torch.sigmoid(outputs).cpu()
            labels = labels.cpu()
            for i in range(len(seq_lens)):
            # match lengths of appended items to truncated length from cnn
                truncated_len = len(probabilities)
                raw_probabilities.append(probabilities[i].numpy())
                all_labels.append(labels[i, :truncated_len].cpu().numpy())
                all_subjects.append(subjects[i][:truncated_len])
                all_runs.append(runs[i][:truncated_len])
                all_timestamps.extend(timestamps[i][:truncated_len].cpu().numpy())
                sequence_lengths.append(seq_lens[i].item())

    print("Checking array lengths before aggregation...")
    '''min_length = min(
        len(all_labels),
        len(raw_probabilities),
        len(sequence_lengths),
        len(all_subjects),
        len(all_runs),
        len(all_timestamps),
    )
    all_labels = all_labels[:min_length]
    raw_probabilities = raw_probabilities[:min_length]
    sequence_lengths = sequence_lengths[:min_length]
    all_subjects = all_subjects[:min_length]
    all_runs = all_runs[:min_length]
    all_timestamps = all_timestamps[:min_length]
    '''

    print("Aggregating results...")
    agg_classes, agg_probs, agg_labels, agg_subs = aggregate_probabilities(
        labels=all_labels,
        probabilities=raw_probabilities,
        sequence_lengths=sequence_lengths,
        all_subjects=all_subjects,
        all_runs=all_runs,
        all_timesteps=all_timestamps
    )

    shared_keys = sorted(agg_labels.keys())
    aggregated_classifications = torch.tensor([agg_classes[key] for key in shared_keys])
    aggregated_probabilities = torch.tensor([agg_probs[key] for key in shared_keys])
    aggregated_labels = torch.tensor([agg_labels[key] for key in shared_keys])
    binarized_labels = (aggregated_labels >= 0.5).long()

    # Generate results DataFrame
    results = pd.DataFrame({
        "Timestep": [key[2] for key in shared_keys],
        "True_Label": [agg_labels[key] for key in shared_keys],
        "Pred_Label": [agg_classes[key] for key in shared_keys],
        "Probability": [agg_probs[key] for key in shared_keys],
        "Run": [key[1] for key in shared_keys],
        "Subject": [agg_subs[key] for key in shared_keys]
    })

    # Add error types
    results["Error_Type"] = "TP"
    results.loc[(results["True_Label"] == 1) & (results["Pred_Label"] == 1), "Error_Type"] = "TP"
    results.loc[(results["True_Label"] == 1) & (results["Pred_Label"] == 0), "Error_Type"] = "FN"
    results.loc[(results["True_Label"] == 0) & (results["Pred_Label"] == 1), "Error_Type"] = "FP"
    results.loc[(results["True_Label"] == 0) & (results["Pred_Label"] == 0), "Error_Type"] = "TN"

    # Metrics
    overall_auc = auc(aggregated_probabilities, aggregated_labels)
    print(f"AUC: {overall_auc:.4f}")

    flat_probs = aggregated_probabilities.view(-1)
    flat_labs = aggregated_labels.view(-1)
    
    overall_f1 = f1(aggregated_classifications, binarized_labels)
    print(f"Overall F1 Score: {overall_f1: .4f}")

    accuracy = (aggregated_classifications == aggregated_labels).float().mean().item() * 100
    print(f"Accuracy: {accuracy: .2f}%")

    flat_classifications = aggregated_classifications.view(-1)
    flat_binarized_labels = binarized_labels.view(-1)
    conf_mat = confusion_matrix(flat_binarized_labels, flat_classifications)
    conf_df = pd.DataFrame(conf_mat / np.sum(conf_mat, axis=1)[:, None], index=["Not MW", "MW"], columns=["Not MW", "MW"])


    plt.figure(figsize=(15,10))
    sn.heatmap(conf_df, annot=True, cmap="coolwarm", vmin = 0, vmax = 1)
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    # replace fold number with fold number for model being evaluated/ best model
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig(f"D:\\Plots\\CONVLSTM_confmatrix_CNNLSTM_12-5_2.png")

    precision = precision_score(flat_binarized_labels, flat_classifications)
    recall = recall_score(flat_binarized_labels, flat_classifications)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    mcc_score = mcc(aggregated_classifications, flat_binarized_labels)
    print("MCC", mcc_score)

    log_probs = aggregated_probabilities.squeeze(-1)
    criterion = nn.BCELoss()
    log_loss = criterion(log_probs, aggregated_labels)
    print(f"Log loss: {log_loss: .4f}")

    print("Last 5 classifications and truth labels:")
    for i in range(-5, 0):
        print(f"Classification: {int(aggregated_classifications[i].item())}, Truth: {int(aggregated_labels[i].item())}")

    for subject in test_data["Subject"].unique():
        plot_err_over_time(results, subject)


if __name__ == '__main__':
    main()