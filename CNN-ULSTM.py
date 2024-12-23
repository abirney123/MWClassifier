# -*- coding: utf-8 -*-
"""
Created on Dec 5 2024

@author: Omar A.Awajan
"""

# -*- coding: utf-8 -*-
"""
Created on OCT 13 2024

@author: Omar A.Awajan
"""

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
from torchvision.ops import sigmoid_focal_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecisionRecallCurve, BinaryConfusionMatrix, BinaryF1Score, BinaryMatthewsCorrCoef, BinaryPrecision, BinaryRecall
from collections import defaultdict, Counter
import json
import matplotlib.pyplot as pl


def splitting_data(df : pd.DataFrame):
    # written by Alaina Birney, adapted by Omar Awajan
    print(f"Splitting Data and stratifying against mw_bin label")
    sub_df = df.groupby(["Subject"])["is_MW"].mean().reset_index()
    print(sub_df)
    sub_df.columns = ["Subject","proportion_mw"]
    sub_df["mw_bin"] = pd.cut(sub_df["proportion_mw"],bins=3,labels=['low','medium','high'])
    train_pairs, test_pairs = train_test_split(sub_df, test_size = 0.5, random_state=0, stratify=sub_df["mw_bin"])

    train_data = pd.merge(df,train_pairs,on=["Subject"])
    test_data = pd.merge(df,test_pairs,on=["Subject"])
    train_data = train_data.drop(columns=["proportion_mw","mw_bin"])
    test_data = test_data.drop(columns=["proportion_mw","mw_bin"])
    return train_data,test_data

def class_counts(dataset: torch.utils.data.Dataset) -> dict:
    labels = defaultdict(int)
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels[label.item()] += 1
    return dict(labels)

# this function extracts features on the window level
def get_label(df :pd.DataFrame) -> int:
    counts = df['is_MW'].value_counts()
    if (len(counts.keys()) == 1):
        return counts.keys()[0]
    
    ratio = counts.iloc[1]/counts.iloc[0]
    if (ratio > 0.50) & (counts.keys()[0] == 1):
        return 1
    return 0



# use full dataset
def sliding_window(df :pd.DataFrame, window_size :int=2500, step_size :int=250):
    start = 0
    df = df.ffill()
    num_columns = df.shape[1]-1
    windows :list[pd.DataFrame]= []
    labels :list[int]= []
    while start < len(df):
        end = start + window_size
        window = df.iloc[start:end]
        label = get_label(window)
        if True:
            labels.append(label)
            window= window.drop(columns=['is_MW'])
            if len(window) < window_size:
                padding = pd.DataFrame(np.zeros((window_size - len(window), num_columns)))
                padding.columns = window.columns
                window = pd.concat([window, padding], ignore_index=True,axis=0)
                windows.append(window)
                break
        
            windows.append(window)
            start += step_size

    return windows,labels

def sliding_windows_stager(df :pd.DataFrame, window_size :int=2500, step_size :int=250):
    # group by subject ID and run_num
    # send each unique ( SubjectID + run_num ) combo to sliding window function
    # take the list of windows + labels and concat them to a final list
    final_windows :list[pd.DataFrame] = []
    final_labels :list[int] = []
    subjects_split = df["Subject"].value_counts().index
    for subject in subjects_split:
        print(f"Getting {subject} Data")
        runs = df[df["Subject"] == subject]["run_num"].value_counts().index
        for run in runs:
            if (df[ (df["Subject"] == subject) & (df["run_num"] == run) ]['is_MW'] == 1).any():
                pages = df[ (df["Subject"] == subject) & (df["run_num"] == run) ]["page_num"].value_counts().index
                for page in pages:
                    temp_df = df[ (df["Subject"] == subject) & (df["run_num"] == run) & (df["page_num"] == page)]
                    if (temp_df['is_MW'] == 1).any():
                        print(f"Getting {page} Data")
                        temp_df = temp_df.drop(columns=["Subject", "run_num", "page_num"])
                        windows, labels = sliding_window(temp_df, window_size=2500, step_size=250)
                        for i in range(len(windows)):
                            final_windows.append(windows[i])
                            final_labels.append(labels[i])
    final_windows, final_labels =to_torch(final_windows,final_labels)
    return final_windows,final_labels


def to_torch(windows :list[pd.DataFrame], labels :list[int], window_size :int=2500):
    #num_columns = windows[0].shape[1]-1
    print(windows[0].columns)
    dim = window_size*6
    final_windows :list[torch.tensor] = []
    for window in windows:
        window = window.values.flatten().reshape(1,dim)
        final_windows.append(tensor(window,dtype=torch.float32))
    final_labels = tensor(labels, dtype=torch.int)

    return final_windows, final_labels


def data_loader(filepath :str, window_size :int):
    training_windows = []
    training_labels = []
    testing_windows = []
    testing_labels = []
    # Load dataset and drop irrelevant columns
    print(f"Reading the dataset")
    df :pd.DataFrame = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0','tSample',  'sample_id','tSample_normalized']) #  'Unnamed: 0','run_num', 'page_num',
    print(df.columns)

    train_data, test_data = splitting_data(df)
    print(train_data.shape)
    print(test_data.shape)
    training_windows, training_labels = sliding_windows_stager(train_data,window_size=window_size)
    testing_windows, testing_labels = sliding_windows_stager(test_data,window_size=window_size)

    count = np.bincount(training_labels)
    print(count)
    weights = 1.0 / torch.tensor(count,dtype=torch.float)
    print(weights)
    sample_weights  = weights[training_labels]
    print(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = MWDataSet(training_windows, training_labels)
    test_dataset = MWDataSet(testing_windows, testing_labels)
 
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader



class MWDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# CNN
class CNNModel(nn.Module):
    def __init__(self, size=17500):
        super(CNNModel, self).__init__()

        # First 1D Convolution layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=0, stride=7)
        conv1_out_size = ((size - 7) // 7) + 1
        
        # Second 1D Convolutional layer
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=0, stride=2)
        # conv2_out_size = ((conv1_out_size - 5) // 2) + 1
        
        # Third 1D Convolutional layer
        #self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=2)
        #conv3_out_size = ((conv2_out_size - 3) // 2) + 1
        # Fully connected layers
        self.fc1 = nn.Linear(32 * conv1_out_size,512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 1)  # Output layer

        self.Gelu = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.Gelu(out)
        
        out = out.view(out.size(0), -1)  # Flatten the tensor
        
        # Forward pass through fully connected layers
        out = self.fc1(out)
        out =self.Gelu(out)
        out = self.fc2(out)
        out = self.Gelu(out)
        out = self.fc3(out)
        out = self.Gelu(out)
        out = self.fc4(out)
        out = self.Gelu(out)
        out = self.fc5(out)

        return out
        

#  Windows are Fully Flattened -> LSTM processes each timestep individually with Gelu
class ConvLSTMModel(nn.Module):
    def __init__(self, size=17500, output_dim=1):
        super(ConvLSTMModel, self).__init__()

        # First 1D Convolution layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=6, padding=0, stride=6)
        conv1_out_size = ((size - 7) // 7) + 1
        # Second 1D Convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, stride=6, padding=0)
        conv2_out_size = ((conv1_out_size - 6) // 6) + 1
        
        # Third 1D Convolutional layer
        #self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=2)
        #conv3_out_size = ((conv2_out_size - 3) // 2) + 1
        # Fully connected layers
          
        # LSTM Model
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=5,
            batch_first=True
        )

        # Fully connected layer to output a label for each timestep
        self.fc1 = nn.Linear(128,512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 1)  # Output layer

        self.Gelu = nn.GELU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.Gelu(out)
        
        out = self.conv2(out)
        out = self.Gelu(out)
        out = out.permute(0, 2, 1)
    
        # LSTM
        out, _ = self.lstm(out)
        out = out[:, -1, :]  # Last hidden state



        # Forward pass through fully connected layers
        out = self.fc1(out)
        out =self.Gelu(out)
        out = self.fc2(out)
        out = self.Gelu(out)
        out = self.fc3(out)
        out = self.Gelu(out)
        out = self.fc4(out)
        out = self.Gelu(out)
        out = self.fc5(out)

        return out




def CNNULSTM() -> None:
    # Select GPU for training and testing if available
    hw_device = device("cuda" if cuda.is_available() else "cpu")
    print(f"Using device: {hw_device}")
    results :dict = {}
   
    if True:
        window_size = 2500
        # Initialize model, loss, optimizer, and epochs
        model = ConvLSTMModel(size=window_size*6)
        model.to(hw_device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters())
        epochs = 64
        batch = 32

        # Initialize metrics for evaluation
        accuracy = BinaryAccuracy().to(hw_device)
        precision = BinaryPrecision().to(hw_device)
        recall = BinaryRecall().to(hw_device)
        f1score = BinaryF1Score().to(hw_device)
        auroc = BinaryAUROC().to(hw_device)
        confusionmatrix = BinaryConfusionMatrix().to(hw_device)
        corr_coef = BinaryMatthewsCorrCoef().to(hw_device)
        log_losses = []
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        roc_aucs = []
        file_path = r"D:\all_subjects_data_no_interpolation.csv"
        # file_path = r"Z:\Mindless Reading\neuralnet_classifier\all_subjects_interpolated.csv"
        # file_path = r"D:\train_balanced_newstrat_10k.csv"
        #file_path = r"D:\all_subjects_interpolated.csv"
        train_loader, test_loader = data_loader(file_path,window_size=window_size)

        # Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(hw_device)
                labels = labels.to(hw_device)

                optimizer.zero_grad()
                outputs = model(inputs).view(-1)

                # loss = criterion(outputs, labels.float())
                loss = sigmoid_focal_loss(outputs, labels.float(), alpha=1, gamma=1.5, reduction='mean')
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            log_losses.append(avg_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")
            if avg_loss < best_loss:
                print(f"Validation loss improved from {best_loss:.6f} to {avg_loss:.6f}. Saving model...")
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")  # Save the best model
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Testing Loop
        # Initialize metric tracking lists
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auroc_scores = []
        conf_matrices = []
        corr_coefs = []
        log_losses = []
        model.eval()
        all_outputs = []
        all_labels = []

        # Accumulate metrics across test epochs
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(hw_device)
                labels = labels.to(hw_device)
        
                # Model predictions
                outputs = model(inputs)
        
                # Collect outputs and labels for batch
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

            # Concatenate all outputs and labels across batches
            all_outputs = torch.cat(all_outputs).to(hw_device)
            all_labels = torch.cat(all_labels).to(hw_device)
    
            # Binarize predictions for classification metrics
            predicted_classes = (all_outputs >= 0.5).long().view(-1)
            true_classes = all_labels.view(-1)
    
            # Calculate and store metrics
            test_accuracy = accuracy(true_classes, predicted_classes).item()
            test_auroc = auroc(true_classes, predicted_classes).item()
            test_precision = precision(true_classes, predicted_classes).item()
            test_recall = recall(true_classes, predicted_classes).item()
            test_f1score = f1score(true_classes, predicted_classes).item()
            test_conf_matrix = confusionmatrix(true_classes, predicted_classes)
            test_corrcoef = corr_coef(true_classes, predicted_classes).item()
    
            # Calculate Log Loss (optional, assuming criterion exists)
            loss = criterion(all_outputs.squeeze(-1), all_labels.float()).item()

            # Append each metric for later plotting
            accuracy_scores.append(test_accuracy)
            precision_scores.append(test_precision)
            recall_scores.append(test_recall)
            f1_scores.append(test_f1score)
            auroc_scores.append(test_auroc)
            conf_matrices.append(test_conf_matrix)
            corr_coefs.append(test_corrcoef)
            log_losses.append(loss)

        # Display and store collected metrics
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUROC: {test_auroc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1score:.4f}")
        print(f"Test Confusion Matrix: \n{test_conf_matrix}")
        print(f"Test Correlation Coefficient: {test_corrcoef:.4f}")
        print(f"Test Log Loss: {loss:.4f}")
        
        results[window_size]= {
        "Test Accuracy":test_accuracy,
        "Test AUROC":test_auroc,
        "Test Precision":test_precision,
        "Test Recall":test_recall,
        "Test F1-Score":test_f1score,
        "Test Confusion Matrix":test_conf_matrix,
        "Test Correlation Coefficient":test_corrcoef,
        "Test Log Loss":loss
        }
    
    # Plotting the collected metrics over epochs/iterations
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.plot(accuracy_scores, label='Accuracy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(precision_scores, label='Precision', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Test Precision')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(recall_scores, label='Recall', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Test Recall')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(f1_scores, label='F1 Score', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Test F1 Score')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(auroc_scores, label='AUROC', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.title('Test AUROC')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(log_losses, label='Log Loss', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('Test Log Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main()->None:
    CNNULSTM()


if __name__ == '__main__':
    main()