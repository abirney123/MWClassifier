# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:02:58 2024

@author: abirn

Test best model resulting from cv
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
from sklearn.preprocessing import StandardScaler
from torchmetrics import F1Score, AUROC, MatthewsCorrCoef
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sn
import joblib


#%% hyperparams
input_size = 6 
hidden_size = 256 
num_layers = 2 # more than 3 isn't usually valuable
output_size = 1 # how many values to predict for each timestep
dropout_percent = .3
batch_size = 128
step_size = 250
sequence_length = 2000

#%% functions
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_p = dropout_percent):
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
#%% load data and scaler

file_path = "E:\\MW_Classifier_Data\\test.csv"
test_data = load_data(file_path)

# initialize scaler
# replace cv0 in path with str corresponding to cv run

scaler = joblib.load("C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\cv0_models\\Scaler_fold_12.pk1")


#%% Load saved model 
print(torch.__version__)            # Check PyTorch version
print(torch.cuda.is_available())    # Check if PyTorch detects CUDA
print(torch.cuda.device_count()) 
# check if GPU available and move model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# replace cv0 in path with str corresponding to cv run
model_path = "C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\cv0_models\\LSTM_fold_12.pth"
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(model_path))

#%% apply scaler, define dataset and dataloader
columns_to_scale = ["LX", "LY", "RX", "RY"]
test_scaled = test_data.copy()
test_scaled.loc[:,columns_to_scale] = scaler.transform(test_data[columns_to_scale])

test_dataset = WindowedTimeSeriesDataset(test_data, sequence_length, step_size)

testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, 
                        collate_fn = add_padding)

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

#%%

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
plt.title("Confusion Matrix: LSTM CV Fold 12")

plt.savefig(f"C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\plots\\cv0\\LSTM_confmatrix.png")

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