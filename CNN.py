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
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecisionRecallCurve, BinaryConfusionMatrix, BinaryF1Score, BinaryMatthewsCorrCoef, BinaryPrecision, BinaryRecall
from collections import defaultdict, Counter



def class_counts(dataset: torch.utils.data.Dataset) -> dict:
    labels = defaultdict(int)
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels[label.item()] += 1
    return dict(labels)

def combine_records(df :pd.DataFrame) ->pd.DataFrame:
    # written by Omar Awajan, adapted by Alaina Birney
    # omar: I admit I do not test my code
    # Alain: I am aware!
    # .....
    # Alaina: this is how I changed it. Sure thing let me just grab my headphones
    # create mask where True means curr row is different from previous row
    # across all feature columns
    mask = (
        (df['LX'] != df['LX'].shift())
    & (df['LY'] != df['LY'].shift())
    & (df['RX'] != df['RX'].shift())
    & (df['RY'] != df['RY'].shift())
    & (df["LPupil"] != df["LPupil"].shift())
    & (df["RPupil"] != df["RPupil"].shift())
        )
    # Group duplicates
    df['combined'] = mask.cumsum()
    # filter to get the first occurance of each combined group
    df_filtered = df.loc[df.groupby("combined").head(1).index]
    # drop combined column
    df_filtered = df_filtered.drop(columns=["combined"]).reset_index()
    return df_filtered
 
# this is how I changed it. Sure thing let me just grab my headphones

def get_label(df :pd.DataFrame) -> int:
    counts = df['is_MW'].value_counts()
    if (len(counts.keys()) == 1):
        return counts.keys()[0]
    ratio = counts.iloc[1]/counts.iloc[0]
    if (ratio > 0.10) & (counts.keys()[0] == 1):
        return 1
    return 0


def sliding_window(df :pd.DataFrame, window_size :int=2500, step_size :int=250):
    start = 0
    # forward propagation to fill NaN values
    df = df.ffill()
    num_columns = df.shape[1]-1
    windows :list[pd.DataFrame]= []
    labels :list[int]= []
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

    labels = tensor(labels, dtype=torch.int)
    windows = windows

    return windows,labels

def duplicate(dataset) -> torch.utils.data.Dataset:
    """Function to balance dataset by duplicating minority class instances."""
    counts = class_counts(dataset)
    max_count = max(counts.values())
    new_features = []
    new_labels = []
    for i in range(len(dataset)):
        feature, label = dataset[i]
        new_features.append(feature)
        new_labels.append(label)
        num_duplicates = max_count - counts[label.item()]
        if num_duplicates > 0:
            for _ in range(num_duplicates):
                new_features.append(feature.clone())
                new_labels.append(label.clone())
    new_features = torch.stack(new_features)
    new_labels = torch.tensor(new_labels)
    return MWDataSet(features=new_features, labels=new_labels)

class MWDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNModel(nn.Module):
    def __init__(self, size=17500):
        super(CNNModel, self).__init__()

        # First 1D Convolution layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=0, stride=5)
        conv1_out_size = ((size - 5) // 5) + 1
        
        # Second 1D Convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=0, stride=2)
        conv2_out_size = ((conv1_out_size - 5) // 2) + 1
        
        # Third 1D Convolutional layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=2)
        conv3_out_size = ((conv2_out_size - 3) // 2) + 1

        # Fully connected layers
        self.fc1 = nn.Linear(128 * conv3_out_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        #print(f"input shape: {x.shape}")
        out = self.conv1(x)
        #print(f"post conv input shape: {x.shape}")
        out = F.relu(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        #print(f"flattened input shape: {x.shape}")        

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

        return out

'''
    ['Unnamed: 0', 'tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil',
       'page_num', 'run_num', 'is_MW', 'sample_id', 'Subject',
       'tSample_normalized']
'''
def data_loader(filepath :str):

    # Load dataset and drop irrelevant columns
    df :pd.DataFrame = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0', 'page_num', 'tSample', 'sample_id','tSample_normalized']) # 'run_num',
    print(df.columns)
    #split DataSet by Subject
    #trainSubjects, testSubjects = lstm_VC.split_data(df)
    
    subjectList :list[str] = df['Subject'].value_counts().keys()
    totalSize = len(subjectList)
    trainSize = int(0.8 * totalSize)
    trainSubjects :pd.DataFrame = df[df['Subject'].isin(subjectList[:trainSize])]
    testSubjects :pd.DataFrame = df[df['Subject'].isin(subjectList[trainSize:])]
    trainSubjects.drop(columns=['Subject'],inplace=True)
    testSubjects.drop(columns=['Subject'], inplace=True)
    print(trainSubjects.shape)
    print(testSubjects.shape)
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
    optimizer = Adam(model.parameters())
    epochs = 64
    batch = 10

    # init metrics for model evaluation
    accuracy = BinaryAccuracy()
    auroc = BinaryAUROC()
    precision = BinaryPrecision()
    confusionmatrix = BinaryConfusionMatrix()
    f1score = BinaryF1Score()
    corr_coef = BinaryMatthewsCorrCoef()
    precision_recallcurve = BinaryPrecisionRecallCurve()
    recall = BinaryRecall()


    # prepare dataset for model learning
    file_path = r"Z:\Mindless Reading\neuralnet_classifier\all_subjects_interpolated.csv" # "D:\\all_subjects_data_no_interpolation.csv"
    training_windows, training_labels, testing_windows, testing_labels = data_loader(file_path)
    
    train_dataset = MWDataSet(features=training_windows, labels=training_labels)
    test_dataset = MWDataSet(features=testing_windows, labels=testing_labels)
    
    train_dataset = duplicate(train_dataset)
    test_dataset = duplicate(test_dataset)

    print(f"Training Dataset: {class_counts(train_dataset)}")
    print(f"Testing Dataset: {class_counts(test_dataset)}")
    
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
    

    # calculate model performance
    acc = accuracy(true_classes, predicted_classes)
    au_roc = auroc(true_classes, predicted_classes)
    prec = precision(true_classes, predicted_classes)
    rec = recall(true_classes, predicted_classes)
    f1_score = f1score(true_classes, predicted_classes)
    conf_mat = confusionmatrix(true_classes, predicted_classes)
    corrcoef = corr_coef(true_classes, predicted_classes)
    prec_recallcurve = precision_recallcurve(true_classes, predicted_classes)
    
    # print model performance results 
    print(f'Accuracy: {acc:.4f}')
    print(f'AUROC: {au_roc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-Score: {f1score:.4f}')
    print(f'confusion matrix: {conf_mat}')
    print(f'correlation coff: {corrcoef}')
    print(f'precision recall curve: {prec_recallcurve}')



def main()->None:
    CNN()
    

if __name__ == '__main__':
    main()