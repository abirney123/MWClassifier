from ast import Try
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch import tensor,device,cuda
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1Score, AUROC
import torch.nn.functional as F
from torch.optim import Adam,AdamW
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_label(df :pd.DataFrame) -> bool:
    counts = df['is_MW'].value_counts()
    if (len(counts.keys()) == 1):
        return bool(counts.keys()[0])
    ratio = counts.iloc[1]/counts.iloc[0]
    if (ratio > 0.10) & (counts.keys()[0] == 1):
        return 1
    return 0


def sliding_window(df :pd.DataFrame, window_size :int=2000, step_size :int=500):
    start = 0
    # forward propagation to fill NaN values
    df = df.ffill()
    num_columns = df.shape[1]-1
    windows :list[pd.DataFrame]= []
    labels :list[bool]= []
    while start < len(df):
        end = start + window_size
        window = df.iloc[start:end]
        label = get_label(window)
        labels.append(label)
        window= window.drop(columns=['is_MW'])
        if len(window) < window_size:
            padding = pd.DataFrame(np.zeros((window_size - len(window), num_columns)))
            padding.columns = window.columns
            window = pd.concat([window, padding], ignore_index=True,axis=0)
            window = window.values.flatten().reshape(1,window_size*num_columns)
            window = tensor(window,dtype=torch.float32)
            windows.append(window)
            break

        window = window.values.flatten().reshape(1,window_size*num_columns)
        window = tensor(window,dtype=torch.float32)
        
        windows.append(window)
        start += step_size

    labels = tensor(labels, dtype=torch.bool)
    windows = windows

    return windows,labels


class MWDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNModel(nn.Module):
    def __init__(self, size=12000):
        super(CNNModel, self).__init__()

        # 1D Convolution layer with a single layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=6, padding=0)
        
        self.final_size = (size - 6) + 1  # Final size after conv layer
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * self.final_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)  # Output layer for binary classification

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        
        # Forward pass through fully connected layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        # Apply sigmoid activation for binary classification
        # out = torch.sigmoid(out)

        return out

'''
    ['Unnamed: 0', 'tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil',
       'page_num', 'run_num', 'is_MW', 'sample_id', 'Subject',
       'tSample_normalized']
'''
def data_loader(filepath :str):

    # Load dataset and drop irrelevant columns
    df :pd.DataFrame = pd.read_csv(filepath)
    df.drop(columns=['Unnamed: 0', 'page_num', 'tSample','run_num','sample_id','tSample_normalized'], inplace=True)
    
    #split DataSet by Subject
    subjectList :list[str] = df['Subject'].value_counts().keys()
    totalSize = len(subjectList)
    trainSize = int(0.8 * totalSize)
    trainSubjects :pd.DataFrame = df[df['Subject'].isin(subjectList[:trainSize])]
    testSubjects :pd.DataFrame = df[df['Subject'].isin(subjectList[trainSize:])]
    trainSubjects.drop(columns=['Subject'],inplace=True)
    testSubjects.drop(columns=['Subject'], inplace=True)
    
    training_windows, training_labels = sliding_window(trainSubjects)
    testing_windows, testing_labels = sliding_window(testSubjects)
    
    return training_windows, training_labels, testing_windows, testing_labels 


def CNN()->None:
    # select GPU for training and testing if avaialble
    hw_device = device("cuda" if cuda.is_available() else "cpu")
    print(f"Using device: {hw_device}")
    # init model parameters
    model = CNNModel()
    model.to(hw_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 64
    batch = 10

    # prepare dataset for model learning
    file_path = "D:\\all_subjects_data_no_interpolation.csv"
    training_windows, training_labels, testing_windows, testing_labels = data_loader(file_path)
    
    train_dataset = MWDataSet(features=training_windows, labels=training_labels)
    test_dataset = MWDataSet(features=testing_windows, labels=testing_labels)
    
    train_loader = DataLoader(train_dataset,batch_size=batch,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch,shuffle=False)    

    # create model
    model = CNNModel()
    model.to(hw_device)

    # training Loop
    loss_vals: list = []
    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0  
    
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(hw_device)
            labels = labels.to(hw_device)
        
            optimizer.zero_grad()
            output = model(inputs)
        
            loss = criterion(output, labels.view(-1, 1).float())
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, step {i + 1}/{len(train_loader)}, loss {loss.item()}")

        avg_loss = running_loss / len(train_loader) 
        loss_vals.append(avg_loss) 
        print(f'Epoch [{epoch + 1}/{epochs}], average loss: {avg_loss:.4f}') 

    # testing loop 
    correct = 0
    total = 0
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(test_loader):
        
            inputs = inputs.to(hw_device)

            outputs = model(inputs)

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    predicted_classes = (all_outputs >= 0.5).long().view(-1).cpu().numpy()
    true_classes = all_labels.view(-1).cpu().numpy()
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')




def main()->None:
    CNN()
    

if __name__ == '__main__':
    main()