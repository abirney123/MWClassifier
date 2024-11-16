# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:46:18 2024

@author: abirn
"""

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
all_losses = pd.read_csv("C:/Users/abirn/OneDrive/Desktop/MW_Classifier/focal_grid_search.csv")
print(all_losses.columns)
# where Epoch = 25, find min val loss and print fold

def evaluate_loss_trend(df):
    """
    Evaluate the loss trend over epochs to determine which model had the most
    consistently decreasing loss.
    """
    # get all unique alpha gamma combos (these are unique models)
    param_combos = df[["Alpha", "Gamma"]].drop_duplicates()
    best_model = None
    best_score = float("inf")
    
    # iterate over hyperparam combinations
    for _, params in param_combos.iterrows():
        # get data for this combo
        model_data = df[
            (df["Alpha"] == params["Alpha"]) & 
            (df["Gamma"] == params["Gamma"])
        ]    
        # get difference in loss between consecutive epochs
        loss_diff = model_data["Train_loss"].diff().dropna()
        decreases = (loss_diff < 0).sum() # get the number times that loss
        # decreased over consecutive epochs
        total_epochs = len(loss_diff)
        # get decrease ratio
        if total_epochs > 0:
            decrease_rat = decreases/ total_epochs
        else:
            decrease_rat = 0
        score = 1-decrease_rat
        # update best model if better score found
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score

#%%
best_params, best_score = evaluate_loss_trend(all_losses)
print(f"Best Alpha and Gamma: {best_params}")
print(f"Best Score (1- decrease ratio): {best_score}")
    
#%%
# only ran for 15 epoch
epoch_15_losses = all_losses[all_losses["Epoch"] == 15]
best_params_row = epoch_15_losses.loc[epoch_15_losses["Train_loss"].idxmin()]
best_alpha = best_params_row["Alpha"]
best_gamma = best_params_row["Gamma"]
best_train_loss = best_params_row["Train_loss"]
multiplier = best_params_row["Multiplier"]
print(f"The model with the lowest train loss was found with alpha = {best_alpha}, gamma = {best_gamma}")
print(f"This alpha was calculated by multiplying 1/proportion_mw times {multiplier}")
print(f"Training Loss: {best_train_loss}")

