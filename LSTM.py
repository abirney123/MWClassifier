#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:53:46 2024

@author: Alaina
Use TorchGPU conda environment 

look into incorporating attention mechanism

"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1Score, AUROC
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler


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
    sub_mw_proportions = dfSamples.groupby("Subject")["is_MW"].mean().reset_index()
    # rename cols for clarity
    sub_mw_proportions.columns = ["Subject", "mw_proportion"]
    
    # get unique subject run pairs
    sub_run_pairs = dfSamples[["Subject", "run_num"]].drop_duplicates()
    
    # merge MW proportions with sub run pairs
    sub_run_pairs = pd.merge(sub_run_pairs, sub_mw_proportions, on="Subject")
    
    # split sub run pairs into train and test 
    # shuffle is true here because just shuffling the sub run pairs, not the 
    # time series data
    # stratified split by mw proportion 
    train_pairs, test_pairs = train_test_split(sub_run_pairs,
                                               test_size = .2, random_state=42,
                                               stratify=sub_run_pairs["mw_proportion"])
    
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
    # add padding to match sequence length, convert input to tensor first
    # pad sequence adds zero padding to match largest sequence in batch
    padded_inputs = pad_sequence([inp.clone().detach() if isinstance(inp, torch.Tensor) else torch.tensor(inp) for inp in inputs], batch_first=True)
    padded_labels = pad_sequence([lbl.clone().detach() if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in labels], batch_first=True)
    return padded_inputs, padded_labels

# define custom DataLoader class
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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # LSTM takes input tensor of shape (batch_size, sequence_length, input_size)
        # bidirectional lstm with batch as first dimension
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True, bidirectional=True)
        #self.fc = torch.nn.Linear(hidden_size*2, hidden_size*2)
        #self.relu = nn.ReLU()
        # fc should be hidden size *2 because bidirectional - map hidden state to output size
        self.fc2 = torch.nn.Linear(hidden_size*2, output_size)
    def forward(self,x):
        # x shape: (batch size, sequence length, number of features)
        out, _ = self.lstm(x)
        #out = self.fc(out)
        #out = self.relu(out)
        out = self.fc2(out)
        return out

#%%

#file_path = "/Volumes/brainlab/Mindless Reading/neuralnet_classifier/all_subjects_interpolated_pupil_coord.csv"
file_path = "E:\\MW_Classifier_Data\\all_subjects_interpolated_pupil_coord.csv"
dfSamples = load_data(file_path)

#%%

# train test split
# do subject-wise train test split to ensure model generalizes well to new subjects
train_data, test_data = split_data(dfSamples)

# drop page num, run num, sample id, tSample, subject from train and test data
train_data = train_data.drop(columns=["page_num", "run_num", "sample_id", "tSample", "Subject"])
test_data = test_data.drop(columns=["page_num", "run_num", "sample_id", "tSample", "Subject"])

print(train_data.columns)
print(test_data.columns)

#%%
# create datasets
# set parameters
sequence_length = 4000
step_size = 500
columns_to_scale = ["LX", "LY", "RX", "RY", "tSample_normalized"]
# initialize scaler
scaler = StandardScaler()
# instantiate - scaling applied in dataset class
train_dataset = WindowedTimeSeriesDataset(train_data, sequence_length,
                                          step_size, scaler = scaler, fit_scaler = True, columns_to_scale = columns_to_scale)
test_dataset = WindowedTimeSeriesDataset(test_data, sequence_length,
                                         step_size,scaler = scaler, fit_scaler = False, columns_to_scale = columns_to_scale)
print(len(train_dataset))
# create DataLoaders - no shuffling as this is time series data
# specify num workers?
# use collate fntn to add padding when sequences vary in length (we have more
# samples for some subjects than others) - removing bc no longer necessary i think collate_fn=add_padding
trainloader = DataLoader(train_dataset, batch_size = 64, shuffle=False)
testloader = DataLoader(test_dataset, batch_size = 64, shuffle=False)

"""
for i, (inputs, labels) in enumerate(trainloader):
    print(f"Batch {i}, raw input shape: {[len(inp) for inp in inputs]}")
    print(f"Batch {i}, padded input shape: {inputs.shape}")
    if inputs.shape[1] == 0:  # Check if sequence length is 0
        print(f"Error: Sequence length is 0 in batch {i}")
"""

#%%
# get sense of how imbalanced classes are - change pos weight if necessary!

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
#%%
# sliding window - applied during training data preparation
print(torch.__version__)            # Check PyTorch version
print(torch.cuda.is_available())    # Check if PyTorch detects CUDA
print(torch.cuda.device_count()) 
# check if GPU available and move model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

#%%
del model
del optimizer
del loss


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

input_size = len(train_data.columns)-1 # num features per timestep, num columns in train or test -1 for labels
hidden_size = 128 # can afford to go bigger bc of our dataset size, but lets start here
num_layers = 2 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
num_epochs = 10
lr = .0001
pos_weight = torch.tensor([294780851/2830876]).to(device) # pos weight is ratio of not MW/ MW to give more weight to pos class
# instantiate LSTM and move to GPU
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # BCE with logits bc binary classification
# pos weight to help with class imbalance
optimizer = optim.Adam(model.parameters(), lr=lr)

# train
# initialize list to store loss for each epoch
loss_vals = []
for epoch in range(num_epochs):
    model.train() # put in training mode
    running_loss = 0.0
    for i, (inputs,labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero gradients for this batch
        optimizer.zero_grad()
        # forward prop - dont need signmoid bc included in loss fntn
        outputs = model(inputs)
        # reshape labels to calculate loss
        loss = criterion(outputs, labels.unsqueeze(-1))
        #backprop
        loss.backward()
        optimizer.step()
        
        # accumulate loss
        running_loss += loss.item()
        
        # output stats
        if i % 100 == 0:
            print("Epoch: %d Minibatch %5d loss: %.3f" %(epoch +1, i+1, loss.item()))
            
    # get and store avg loss for this epoch
    epoch_loss = running_loss / len(trainloader)
    loss_vals.append(epoch_loss)
print("Training Complete")

# save the model
save_path = "C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier.pth"
torch.save(model.state_dict(), save_path)
print("Model saved to ", save_path)
#%%
# plot loss and save plot
plt.plot(range(1,num_epochs+1), loss_vals)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
# get datetime for saving fig
curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f"C:\\Users\\abirn\\OneDrive\\Desktop\\MW_Classifier\\plots\\LSTM_loss_{curr_datetime}.png")
plt.show()

    

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

# initialize f1 and auc for binary classification
f1 = F1Score(task="binary")
auc = AUROC(task="binary")

for i, data in enumerate(testloader):
    with torch.no_grad(): # disable gradient calculation
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward prop
        outputs = model(inputs)
        y_probs = torch.sigmoid(outputs)
        # convert probs to predictions with .5 threshold
        y_preds = (y_probs >= .5).float()
        # remove extra dimension from predictions
        y_preds = y_preds.squeeze(-1)
        
        # add predictions and labels to lists
        all_predictions.append(y_preds.cpu())
        all_labels.append(labels.cpu())
        all_probabilities.append(y_probs.cpu())
        

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
# get f1
overall_f1 = f1(all_predictions, all_labels)
print(f"Overall F1 Score: {overall_f1: .4f}")
# take accuracy with a grain of salt bc our classes are imbalanced
accuracy = (correct/total)*100
print(f"Accuracy: {accuracy: .2f}%")

print("Last 5 predictions and truth labels:")
for i in range(-5,0):
    # printing last item in sequence for five sequences of predictions and the corresponding label
    print(f"Prediction: {int(all_predictions[i, -1].item())}, Truth: {int(all_labels[i, -1].item())}")


