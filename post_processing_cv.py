# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 08:01:21 2024

@author: abirn
"""
import pandas as pd
from matplotlib import pyplot as plt

# find best model- use all losses csv to find the fold corresponding to the 
# lowest validation loss for epoch 25
# change cv0 in file path depending on which cv run you're evaluating
all_losses = pd.read_csv("C:/Users/abirn/OneDrive/Desktop/MW_Classifier/all_fold_losses_cv0.csv")
print(all_losses.columns)
# where Epoch = 25, find min val loss and print fold
#%%
epoch_25_losses = all_losses[all_losses["Epoch"] == 25]
best_fold_row = epoch_25_losses.loc[epoch_25_losses["Validation_Loss"].idxmin()]
best_fold_number = best_fold_row["Fold"]
best_fold_val_loss = best_fold_row["Validation_Loss"]
best_fold_train_loss = best_fold_row["Train_Loss"]
print(f"The model with the lowest validation loss was found at fold {best_fold_number}")
print(f"Training Loss: {best_fold_train_loss}")
print(f"Validation Loss: {best_fold_val_loss}")

#%%
# plot average training and val loss over all folds for each epoch
# average training and val loss over folds
avg_losses = all_losses.groupby("Epoch").agg({"Train_Loss": "mean",
                                              "Validation_Loss": "mean"}).reset_index()
# plot for each epoch (1-25)
plt.figure(figsize=(10,6))
plt.plot(avg_losses["Epoch"], avg_losses["Train_Loss"], label="Training Loss")
plt.plot(avg_losses["Epoch"], avg_losses["Validation_Loss"], label="Validation Loss")

# Step 4: Add titles and labels
plt.title("Training and Validation Loss, Averaged Over Folds")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
# change cv0 in path to save fig depending on cv run
plt.savefig("C:/Users/abirn/OneDrive/Desktop/MW_Classifier/plots/cv0/avg_loss_all_folds.png")
plt.show()