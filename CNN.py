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
import json



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
    if (ratio > 0.50) & (counts.keys()[0] == 1):
        return 1
    return 0

def sliding_window(df: pd.DataFrame, window_size: int = 2500, step_size: int = 250):
    start = 0
    # Forward propagation to fill NaN values
    df = df.ffill()
    num_columns = df.shape[1] - 1
    windows: list[pd.DataFrame] = []
    labels: list[int] = []
    mw_count: int = 0
    f_count: int = 0
    while start < len(df):
        end = start + window_size
        window = df.iloc[start:end]
        label = get_label(window)
        if label == 1:
            mw_count += 1
        else:
            f_count += 1
        labels.append(label)
        window = window.drop(columns=['is_MW'])
        
        # Padding
        if len(window) < window_size:
            padding = pd.DataFrame(np.zeros((window_size - len(window), num_columns)))
            padding.columns = window.columns
            window = pd.concat([window, padding], ignore_index=True, axis=0)
            windows.append(window)
            break
        
        windows.append(window)
        start += step_size
    print(f"Original mw_count: {mw_count} - f_count: {f_count}")
    majority_class_label = 0 if f_count > mw_count else 1
    majority_class_count = max(mw_count, f_count)
    minority_class_count = min(mw_count, f_count)
    drop_count = majority_class_count - minority_class_count
    kept_windows = []
    kept_labels = []
    dropped = 0
    for i, label in enumerate(labels):
        if label == majority_class_label and dropped < drop_count:
            dropped += 1
        else:
            kept_windows.append(windows[i])
            kept_labels.append(label)
    print(f"Dropped {dropped} instances from the majority class")
    final_windows: list[torch.Tensor] = []
    for window in kept_windows:
        window = window.values.flatten().reshape(1, window_size * num_columns)
        window = tensor(window, dtype=torch.float32)
        final_windows.append(window)
    
    final_labels = tensor(kept_labels, dtype=torch.int)
    return final_windows, final_labels

def sliding_window_1(df :pd.DataFrame, window_size :int=2500, step_size :int=250):
    start = 0
    # forward propagation to fill NaN values
    df = df.ffill()
    num_columns = df.shape[1]-1
    windows :list[pd.DataFrame]= []
    labels :list[int]= []
    mw_count :int = 0
    f_count :int = 0
    while start < len(df):
        end = start + window_size
        window = df.iloc[start:end]
        label = get_label(window)
        if label == 1:
            mw_count += 1
        else:
            f_count += 1
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
    print(f"mw_count {mw_count} - f_count {f_count}")
    dif = f_count - mw_count
    index :int = 0
    c :int = 0
    while dif >= 0:
        if index >= len(labels):
            index = 0
        if labels[index] == 1:
            sequence = windows[index]
            label = labels[index]
            windows.append(sequence)
            labels.append(label)
        index += 1
        dif -= 1
        c += 2
    print(f"duplications {c}")
    final_windows :list[torch.Tensor] = []
    for window in windows:
        window = window.values.flatten().reshape(1,window_size*num_columns)
        window = tensor(window,dtype=torch.float32)
        final_windows.append(window)
    
    labels = tensor(labels, dtype=torch.int)
    final_windows = final_windows

    return final_windows,labels


def sliding_window_2(df :pd.DataFrame, window_size :int=2500, step_size :int=250):
    start = 0
    df = df.ffill()
    num_columns = df.shape[1]-1
    windows :list[pd.DataFrame]= []
    labels :list[int]= []
    mw_count :int = 0
    f_count :int = 0
    while start < len(df):
        end = start + window_size
        window = df.iloc[start:end]
        label = get_label(window)
        if label == 1:
            mw_count += 1
        else:
            f_count += 1
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
    
    final_windows :list[torch.tensor] = []
    for window in windows:
        window = window.values.flatten().reshape(1,window_size*num_columns)
        window = tensor(window,dtype=torch.float32)
        final_windows.append(window)
    final_labels = tensor(labels, dtype=torch.int)

    return final_windows,labels


def duplicate(dataset :torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    # Function to balance dataset by duplicating minority class instances.
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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=6, padding=0, stride=6)
        conv1_out_size = ((size - 6) // 6) + 1
        
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

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        #out = self.conv2(out)
        #out = F.relu(out)
        
        out = out.view(out.size(0), -1)  # Flatten the tensor
        
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
def data_loader_1(filepath :str, window_size :int):

    # Load dataset and drop irrelevant columns
    print(f"Reading the dataset")
    df :pd.DataFrame = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0', 'page_num', 'tSample', 'sample_id','tSample_normalized','run_num']) # 'run_num',
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
    training_windows, training_labels = sliding_window_2(trainSubjects,window_size=window_size)
    testing_windows, testing_labels = sliding_window_2(testSubjects,window_size=window_size)

    count = np.bincount(training_labels)
    weights = 1.0 / torch.tensor(count,dtype=torch.float)
    sample_weights  = weights[training_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_dataset = MWDataSet(training_windows, training_labels)
    test_dataset = MWDataSet(testing_windows, testing_labels)
 
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader

def data_loader_2(filepath :str,window_size :int):

    # Load dataset and drop irrelevant columns
    print(f"Reading the dataset")
    df :pd.DataFrame = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0', 'page_num', 'tSample', 'sample_id','tSample_normalized','run_num']) # 'run_num',
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
    training_windows, training_labels = sliding_window(trainSubjects, window_size=window_size)
    testing_windows, testing_labels = sliding_window(testSubjects, window_size=window_size)

    count = np.bincount(training_labels)
    weights = 1.0 / torch.tensor(count,dtype=torch.float)
    sample_weights  = weights[training_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    
    train_dataset = MWDataSet(training_windows, training_labels)
    test_dataset = MWDataSet(testing_windows, testing_labels)
 
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=sampler,shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader 


def data_loader(filepath :str):

    # Load dataset and drop irrelevant columns
    print(f"Reading the dataset")
    df :pd.DataFrame = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0', 'page_num', 'tSample', 'sample_id','tSample_normalized','Subject','run_num'])
    print(df.columns)    
    trainSubjects, testSubjects = train_test_split(df, test_size=0.2, random_state=42,stratify=df['is_MW'] )
    # Check resulting shapes
    print(trainSubjects.shape)
    print(testSubjects.shape)
    # Process the training and testing data
    training_windows, training_labels = sliding_window(trainSubjects)
    testing_windows, testing_labels = sliding_window(testSubjects)
    
    return training_windows, training_labels, testing_windows, testing_labels


def CNN() -> None:
    # Select GPU for training and testing if available
    hw_device = device("cuda" if cuda.is_available() else "cpu")
    print(f"Using device: {hw_device}")
    results :dict = {}
   
    for window_size in range(1000,4000,500):
        # Initialize model, loss, optimizer, and epochs
        model = CNNModel(size=window_size*6)
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
        roc_aucs = []
        file_path = r"Z:\Mindless Reading\neuralnet_classifier\all_subjects_interpolated.csv"
        # training_windows, training_labels, testing_windows, testing_labels = data_loader(file_path)
        # train_dataset = MWDataSet(features=training_windows, labels=training_labels)
        # test_dataset = MWDataSet(features=testing_windows, labels=testing_labels)
        # train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
        # train_loader, test_loader = data_loader_2(file_path,window_size=window_size)
        train_loader, test_loader = data_loader_1(file_path,window_size=window_size)
        # Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(hw_device)
                labels = labels.to(hw_device)

                optimizer.zero_grad()
                outputs = model(inputs).view(-1)
            

                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            log_losses.append(avg_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

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

    with open("results.json", 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
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
    CNN()
    

if __name__ == '__main__':
    main()