import os
import time
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader , SubsetRandomSampler
from torchsummary import summary
import pandas as pd
import numpy as np

from model import *
from datasets import RBNSDataset, RNCMPTDataset
# Set the seed for PyTorch
SEED = random.randrange(0,123456789)
print('Seed: ',SEED)
torch.manual_seed(SEED)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createModel(inputShape, classesNum):
    model = DeepSELEX2(inputShape,classesNum).to(device) # NOTE: Set the model we want to run here.
    # summary(model, inputShape)
    return model

def loadTrainTestLoaders(rbns_file_paths, seqs_per_file, batch_size, test_ratio):
    rbns_dataset = RBNSDataset(rbns_file_paths, SEED, seqs_per_file)

    # Shuffle and split the dataset into training and testing sets
    dataset_size = len(rbns_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    
    # Create data loaders for training and testing
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(rbns_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(rbns_dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, test_loader

def trainModel(rbns_file_paths):
    # Hyperparams:
    seqs_per_file = 15000
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    weight_decay= 0.00001
    betas=(0.9,0.999)
    test_ratio=0.3
    patience=3
    warmup_epochs = 5
    print(f'Run with Hyperparams:\nNumber of Seqs: {seqs_per_file}, Batch_size: {batch_size}, Number of Epochs: {num_epochs},\nLearning Rate: {learning_rate}, Weight Decay: {weight_decay}, Betas: {betas}, Test Ratio: {test_ratio}, Patience: {patience}')
    
    print('Load Data!')
    train_loader, test_loader = loadTrainTestLoaders(rbns_file_paths, seqs_per_file, batch_size, test_ratio)
    inputShape, classesNum = train_loader.dataset.getInputShape(),train_loader.dataset.getClassesNum()
    model = createModel(inputShape, classesNum)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay, betas = betas)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs))

    best_loss = float('inf')
    patience_counter = 0
    
    print('Start Training!')
    # Training loop
    for epochs in range(num_epochs):
        print(f'Epoch [{epochs+1}/{num_epochs}]')
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_data, targets in train_loader:
             # Move batch_data and targets to the appropriate device (CPU or GPU) if needed 
            batch_data = batch_data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(batch_data)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the training loss
            train_loss += loss.item() * batch_data.size(0)  # Accumulate the loss

        # Update scheduler for warmup system.
        scheduler.step()

        # Calculate average training loss for the epoch
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, targets in test_loader:
                # Move batch_data and targets to the appropriate device (CPU or GPU) if needed 
                batch_data = batch_data.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(batch_data)

                # Calculate the loss
                loss = criterion(outputs, targets)

                # Update the validation loss
                val_loss += loss.item() * batch_data.size(0)  # Accumulate the loss

        # Calculate average validation loss for the epoch
        val_loss /= len(test_loader.dataset)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model weights
            best_model_weights = model.state_dict()
        else:
            patience_counter += 1
        print(f'Loss: {val_loss}, Patience [{patience_counter}/{patience}] (Best Loss: {best_loss})')
        # Check if early stopping criteria are met
        if patience_counter >= patience:
            print("Early stopping! Validation loss didn't improve in the last", patience, "epochs.")
            break

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    
    return model, inputShape, classesNum

def evaluateModel(model, RNAcompete_sequences_path, RNCMPT_training_path, inputShape, classesNum, output_path = 'output_file.csv'):
    print(f'Start evaluate!')
    # Create the custom dataset
    eval_dataset = RNCMPTDataset(RNAcompete_sequences_path, RNCMPT_training_path, inputShape[0])
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False) # TODO: support multiple sequences at once, batch_size > 1 (GPU can support with best speed at 512 or 1024, could take 15 seconds instead of few minutes)

    # Set the model to evaluation mode
    model.eval()
    # Evaluation loop
    predicts, truths = [], []
    with torch.no_grad():
        total = len(eval_loader)
        count = 0
        for sequences, targets in eval_loader:
            count += 1
            sequences = sequences.to(device)
            sequence = sequences[0] # TODO: support multiple sequences at once, batch_size > 1
                
            # Run sequences in a sliding window, sized 20 (as in the training set)
            window_size = inputShape[0]
            stride = 1
            predicted_columns = np.empty((0, classesNum))

            # Collect all windows in a list
            windows = [sequence[i : i + window_size] for i in range(0, len(sequence) - window_size + 1, stride)]
            # In case of the window size being bigger than the sequences themself, we send just the sequences, it will pad it if needed.
            if len(windows) <= 0:
                windows = [sequence]
            # Convert the list of windows into a single tensor
            windows_tensor = torch.stack(windows)

            # Pass the input through the model to get the predicted scores for all windows
            predicted_matrices = model(windows_tensor)

            # Convert the predictions to a numpy array
            predicted_columns = predicted_matrices.cpu().numpy()

            # Reshape the predictions to get predicted_scores for each window
            predicted_columns = predicted_columns.reshape(-1, classesNum)

            # Calculate the predicted_score using the formula
            column_0 = predicted_columns[:, 0]
            column_3 = predicted_columns[:, 3]
            column_4 = predicted_columns[:, 4]
            
            predicted_score = -np.min(column_0) + np.max(column_3) + np.max(column_4) # TODO: because of the changing amount of files=classes, We prorbably needs to drift the column_3, column_4 to be in the exact range we expecting to get values from.
            predicts.append(predicted_score)

            # Save targets.
            targets = targets.tolist()
            for target in targets:
                truths.append(target[0]) 

            if count % 25000 == 0: # TODO: hardcoded.
                print(f'Score: {predicted_score}, [{count}/{total}]')

    # Open the file in write mode
    with open(output_path, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Convert the predicts list to a list of lists
        predicts_list = [[value] for value in predicts]
        # Write each list as a row in the CSV file
        for row in predicts_list:
            writer.writerow(row)

    # Convert the lists to tensors
    epsilon = 1e-9  # A small value to avoid division by zero
    predicts_tensor = torch.tensor(predicts) + epsilon
    truths_tensor = torch.tensor(truths) + epsilon

    combined_tensor = torch.stack((predicts_tensor, truths_tensor), dim=0)

    # Calculate Pearson correlation
    correlation_matrix = torch.corrcoef(combined_tensor)
    pearson_correlation = correlation_matrix[0, 1]
    
    print('Pearson correlation:', pearson_correlation.item())
    return pearson_correlation.item()

# Helper function to get all the RBP files, in a list.
def find_rbp_files(dir_path, rbp_num):
    rbp_files = [
        f'{dir_path}/RBNS_training/{file}'
        for file in os.listdir(f'{dir_path}/RBNS_training')
        if file.startswith(f'RBP{rbp_num}_') and file.endswith('.seq')
    ]
    # Define a function to extract numeric values from the file name
    def extract_numeric_value(file_name):
        if '_input.seq' in file_name:
            return (0, file_name)  # Sort _input.seq files first
        start_idx = file_name.rfind('_') + 1
        end_idx = file_name.rfind('nM.seq')
        return (int(file_name[start_idx:end_idx]), file_name)
    
    return sorted(rbp_files, key=extract_numeric_value)


# helper function running over all RBPs, and train and evulate each, saving the pearson in a file.
def RunOverAll():
    num_rbns_files = 16
    dir_path = '.'
    with open('pearson.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # write row.
        writer.writerow([f'RBP Num', 'Pearson Correlation', 'Train Time (Seconds)', 'Evalute Time (Seconds)', 'Seed'])
    # Loop through the RBPs and process each one
    for rbp_num in range(1, num_rbns_files + 1):
        rbns_file_paths_rbp = find_rbp_files(dir_path, rbp_num) # Automatically find all the files and send them sorted in the right way.
        RNAcompete_sequences_path_rbp = f"{dir_path}/RNAcompete_sequences.txt"
        RNCMPT_training_path_rbp = f"{dir_path}/RNCMPT_training/RBP{rbp_num}.txt"

        start_time = time.time()

        model, inputShape, classesNum = trainModel(rbns_file_paths_rbp)

        end_time = time.time()
        elapsed_time = end_time - start_time
        train_time = elapsed_time
        print(f"Elapsed Time for training: {elapsed_time} seconds")
        start_time = time.time()
        pearson_correlation = evaluateModel(model, RNAcompete_sequences_path_rbp, RNCMPT_training_path_rbp, inputShape, classesNum, output_path=f'outputs/RBP{rbp_num}_output.csv')
        end_time = time.time()
        elapsed_time = end_time - start_time
        evalute_time = elapsed_time
        print(f"Elapsed Time for evaluating: {elapsed_time} seconds")
        # Open the file in write mode
        with open('pearson.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # write row.
            writer.writerow([f'RBP{rbp_num}', pearson_correlation, train_time, evalute_time, SEED])

if __name__ == '__main__':
    RunOverAll()
            


        