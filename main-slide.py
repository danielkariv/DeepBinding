import sys
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
from tensorboardX import SummaryWriter
from model import *
from datasets import RBNSDataset, RNCMPTDataset
# Set the seed for PyTorch
SEED = random.randrange(0,123456789)
print('Seed: ', SEED)
torch.manual_seed(SEED)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a summary writer
tensorboard_writer = SummaryWriter() # NOTE: See graphs when running: tensorboard --logdir=runs, and then entering the localhost web-server.


def createModel(inputShape, classesNum):
    model = DeepMultiConvModel(inputShape,classesNum).to(device) # NOTE: Set the model we want to run here.
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


class SlidingPipeline:
    def __init__(self):
        print('Run with sliding window pipeline.')
        
    def trainModel(self, rbns_file_paths):
        # Hyperparams:
        seqs_per_file = 15000
        batch_size = 64
        num_epochs = 0
        learning_rate = 1e-3
        weight_decay= 1e-5
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
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay, betas = betas, amsgrad=False)
        # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs))

        best_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        print('Start Training!')
        # Training loop
        for epoch in range(num_epochs):
            print(f'Epoch [{epoch+1}/{num_epochs}]')
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
            
            tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch)
            tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch)

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
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        
        self.model, self.inputShape, self.classesNum = model, inputShape, classesNum

    def predict(self, RNAcompete_sequences_path, output_path = 'output_file.csv'):
        print(f'Start evaluate!')
        # Create the custom dataset
        eval_dataset = RNCMPTDataset(RNAcompete_sequences_path, self.inputShape[0])
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False) # TODO: support multiple sequences at once, batch_size > 1 (GPU can support with best speed at 512 or 1024, could run in seconds instead of few minutes)

        # Set the model to evaluation mode
        self.model.eval()
        # Evaluation loop
        predicts = []
        with torch.no_grad():
            total = len(eval_loader)
            count = 0
            for sequences in eval_loader:
                count += len(sequences)
                sequences = sequences.to(device)
                sequence = sequences[0] # TODO: support multiple sequences at once, batch_size > 1
                
                # Run sequences in a sliding window, sized 20 (as in the training set)
                window_size = self.inputShape[0]
                stride = 1
                predicted_columns = np.empty((0, self.classesNum))

                # Collect all windows in a list
                windows = [sequence[i : i + window_size] for i in range(0, len(sequence) - window_size + 1, stride)]
                # In case of the window size being bigger than the sequences themself, we send just the sequences, it will pad it if needed.
                if len(windows) <= 0:
                    windows = [sequence]
                # Convert the list of windows into a single tensor
                windows_tensor = torch.stack(windows)

                # Pass the input through the model to get the predicted scores for all windows
                predicted_matrices = self.model(windows_tensor)

                # Convert the predictions to a numpy array
                predicted_columns = predicted_matrices.cpu().numpy()

                # Reshape the predictions to get predicted_scores for each window
                predicted_columns = predicted_columns.reshape(-1, self.classesNum)

                # Calculate the predicted_score using the formula
                column_0 = predicted_columns[:, 0]
                column_1 = predicted_columns[:, 1]
                column_2 = predicted_columns[:, 2]
                column_3 = predicted_columns[:, 3]
                column_4 = predicted_columns[:, 4]
                column_5 = predicted_columns[:, 5]

                # TODO: because of the changing amount of files=classes, We prorbably shold drift score using differnet columns than column_3, column_4 to be in the exact range we expecting to get values from.
                predicted_score = -np.min(column_0) + np.max(column_3) + np.max(column_4) 
                predicts.append(predicted_score)

                if count % 25000 == 0: # TODO: hardcoded (how many to pass before printing progress).
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

        return predicts

import torch.nn.functional as F

def predict(model, RNAcompete_sequences_path, inputShape, classesNum, output_path = 'output_file.csv'):
    print(f'Start evaluate!')
    # Create the custom dataset
    eval_dataset = RNCMPTDataset(RNAcompete_sequences_path, inputShape[0])
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False) # TODO: support multiple sequences at once, batch_size > 1 (GPU can support with best speed at 512 or 1024, could run in seconds instead of few minutes)

    # Set the model to evaluation mode
    model.eval()
    # Evaluation loop
    predicts = []
    with torch.no_grad():
        total = len(eval_loader)
        count = 0
        for sequences in eval_loader:
            count += len(sequences)
            sequences = sequences.to(device)

            # Create a tensor from the padded sequences
            padded_sequences_tensor = torch.stack(padded_sequences)

            # Pass the entire batch of sequences through the model
            predicted_matrices = model(padded_sequences_tensor)

            # Convert the predictions to a numpy array
            predicted_columns = predicted_matrices.cpu().numpy()
            print(predicted_columns, predicted_columns.shape)

            # Calculate the predicted_scores for each item in the batch
            batch_scores = []
            for i in range(len(sequences)):
                column_0 = predicted_columns[i, 0]
                column_1 = predicted_columns[i, 1]
                column_2 = predicted_columns[i, 2]
                column_3 = predicted_columns[i, 3]
                column_4 = predicted_columns[i, 4]
                column_5 = predicted_columns[i, 5]

                # TODO: Calculate predicted_score based on your specific formula
                # For example:
                predicted_score = -np.min(column_0) + np.max(column_3) + np.max(column_4)
                batch_scores.append(predicted_score)

            predicts.extend(batch_scores)

            if count % 25000 == 0: # TODO: hardcoded (how many to pass before printing progress).
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

    return predicts

def pearson_compare(predicts, truths):
    # Convert the lists to tensors
    epsilon = 1e-9  # A small value to avoid division by zero
    predicts_tensor = torch.tensor(predicts) + epsilon
    truths_tensor = torch.tensor(truths) + epsilon
    print(len(predicts_tensor), len(truths_tensor))
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
        sp = SlidingPipeline()
        sp.trainModel(rbns_file_paths_rbp)
        end_time = time.time()
        elapsed_time = end_time - start_time
        train_time = elapsed_time
        print(f"Elapsed Time for training: {elapsed_time} seconds")

        start_time = time.time()
        predicts = sp.predict(RNAcompete_sequences_path_rbp, output_path=f'outputs/RBP{rbp_num}_output.csv')
        end_time = time.time()
        elapsed_time = end_time - start_time
        evalute_time = elapsed_time
        print(f"Elapsed Time for predicts: {elapsed_time} seconds")

        # TODO: used for debug. can be disabled when doing real runs.
        with open(RNCMPT_training_path_rbp, 'r') as f:
            targets = [float(line.strip()) for line in f.readlines()]
        pearson_correlation = pearson_compare(predicts, targets)
        tensorboard_writer.add_scalar('Pearson_Correlation', pearson_correlation, rbp_num)
        # Open the file in write mode
        with open('pearson.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # write row.
            writer.writerow([f'RBP{rbp_num}', pearson_correlation, train_time, evalute_time, SEED])
        
def receiveArgs():
    args_count = len(sys.argv) - 1  # Ignore the 1st, it is the script name.
    if args_count >= 6 and args_count <= 8: # includes: ofile, RNCMPT, input, RBNS1, RBNS2, .. RBNS5
        ofile = sys.argv[1]
        RNCMPT = sys.argv[2]
        RBNS = sys.argv[3:] # Should be sorted by how it given(for example: input, 5nm, ... ,3200nm)
        print(ofile, RNCMPT, RBNS)
        return ofile, RNCMPT, RBNS
    else:
        print("No correct arguments provided.")
        sys.exit(1) # Exit with an error status

if __name__ == '__main__':
    RunOverAll()
    tensorboard_writer.close()
    exit()
    ## Commend like this need to be written (basically as we instructed):
    ## NOTE COMMEND that runs it.
    ## 'python main.py test.txt datasets\RNAcompete_sequences.txt datasets\RBNS_training\RBP1_input.seq datasets\RBNS_training\RBP1_5nm.seq datasets\RBNS_training\RBP1_20nm.seq datasets\RBNS_training\RBP1_80nm.seq datasets\RBNS_training\RBP1_320nm.seq datasets\RBNS_training\RBP1_1300nm.seq'
    ofile, RNCMPT, RBNS = receiveArgs()

    start_time = time.time()

    model, inputShape, classesNum = trainModel(RBNS)

    end_time = time.time()
    elapsed_time = end_time - start_time
    train_time = elapsed_time
    print(f"Elapsed Time for training: {elapsed_time} seconds")

    start_time = time.time()

    predicts = predictSlidingWindow(model, RNCMPT, inputShape, classesNum, output_path=ofile)

    end_time = time.time()
    elapsed_time = end_time - start_time
    evalute_time = elapsed_time
    print(f"Elapsed Time for predicts: {elapsed_time} seconds")