#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:53:46 2024

@author: Alaina
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1Score
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
    # get subjects
    subjects = dfSamples["Subject"].unique()
    # split subjects into train and test 
    train_subjects, test_subjects = train_test_split(subjects, test_size = .2, random_state=42)
    # split data based on subjects
    train_data = dfSamples[dfSamples["Subject"].isin(train_subjects)]
    test_data = dfSamples[dfSamples["Subject"].isin(test_subjects)]
    return train_data, test_data

# define custom DataLoader class
class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=4000, step_size=1000):
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
        self.features = data.drop(columns=["is_MW"]).values
        self.labels = data["is_MW"].values
    
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # get start and end of sliding window
        start_idx = idx * self.step_size
        end_idx = start_idx + self.sequence_length
        
        # get sequence of features
        x = self.features[start_idx:end_idx]
        # get corresponding labels (same length as sequence because many to many)
        y = self.labels[start_idx:end_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# define LSTM class

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # LSTM takes input tensor of shape (batch_size, sequence_length, input_size)
        # bidirectional lstm with batch as first dimension
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True, bidirectional=True)
        # fc should be hidden size *2 because bidirectional - map hidden state to output size
        self.fc = torch.nn.Linear(hidden_size*2, output_size)
    def forward(self,x):
        # x shape: (batch size, sequence length, number of features)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

#%%

file_path = "/Volumes/brainlab/Mindless Reading/neuralnet_classifier/all_subjects_interpolated_pupil_coord.csv"
dfSamples = load_data(file_path)

# train test split
# do subject-wise train test split to ensure model generalizes well to new subjects
train_data, test_data = split_data(dfSamples)


#%%
# create datasets
# set parameters
sequence_length = 4000
step_size = 1000
# instantiate
train_dataset = WindowedTimeSeriesDataset(train_data, sequence_length, step_size)
test_dataset = WindowedTimeSeriesDataset(test_data, sequence_length, step_size)
# create DataLoaders - no shuffling as this is time series data
# specify num workers?
trainloader = DataLoader(train_dataset, batch_size = 128, shuffle=False)
testloader = DataLoader(test_dataset, batch_size = 128, shuffle=False)

#%%
# TBTT - pyTorch does automatically when backprop over truncated sequences
# sliding window - applied during training data preparation

# check if GPU available and move model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
# set parameters
input_size = len(train_data.columns)-1 # num features per timestep, num columns in train or test -1 for labels
hidden_size = 128 # can afford to go bigger bc of our dataset size, but lets start here
num_layers = 1 # more than 3 isn't usually valuable, starting with 1
output_size = 1 # how many values to predict for each timestep
num_epochs = 10
lr = .001
# instantiate LSTM and move to GPU
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCEWithLogitsLoss() # BCE with logits bc binary classification
optimizer = optim.Adam(model.parameters(), lr=lr)

# train
# this is where truncated backprop through time will need to be implemented
for epoch in range(num_epochs +1):
    model.train() # put in training mode
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
        
        # output stats
        if i % 100 == 0:
            print("Epoch: %d Minibatch %5d loss: %.3f" %(epoch +1, i+1, loss.item()))
print("Training Complete")

# save the model
save_path = "/Users/lu/Desktop/MWClassifier"
torch.save(model.state_dict(), save_path)
    
#%% Incomplete
# evaluate on the test set
correct = 0
total = 0
model.eval() # put in evaluation mode

for data in testloader:
    with torch.no_grad():
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward prop
        outputs = model(inputs)
        predictions = torch.argmax(outputs.data, 1)
        
        # update total and correct
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        print("f1 score")
        f1 = F1Score(num_classes=2) # binary classification
        score = f1(predictions, labels)



