import os
import time
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
# Imports our codebase.
from src.Datasets import RBNSDataset, RNCMPTDataset
from src.Models import *

# Set the seed for PyTorch
torch.manual_seed(42)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Const variables (for debug):
TRAINING_PRINT_EVERY_EPOCHS = 1
TRAINING_PRINT_EVERY_BATCHES = -1
EVALUATION_PRINT_EVERY_BATCHES = -1

def createModel():
    model = ModelV4().to(device) # NOTE: Set the model we want to run here.
    # summary(model, (42, 4))
    return model

def trainModel(model, rbns_file_paths, seqs_per_file, batch_size, num_epochs, learning_rate):
    print('Load Data!')
    # NOTE: It loads to memory the seqs. it takes around 20gb on RAM, so run it on a system that can process it.
    #       We could process it by loading when needed a row from it but it is really slow.
    rbns_dataset = RBNSDataset(rbns_file_paths,seqs_per_file) 
    data_loader = DataLoader(rbns_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Start Training!')
    # Training loop
    total_batches = len(data_loader)
    for epoch in range(num_epochs):
        batch_counter = 0  # Counter to track the current batch index within the epoch
        for batch_data, targets in data_loader:
            # Move batch_data to the appropriate device (CPU or GPU) if needed 
            # NOTE: Remember to call .to(device) on any input data you pass to the model during inference as well.
            batch_data = batch_data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(batch_data)

            # Forward pass
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Calculate progress and print
            progress = batch_counter / total_batches * 100
            if TRAINING_PRINT_EVERY_BATCHES >= 0 and (batch_counter + 1) % TRAINING_PRINT_EVERY_BATCHES == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_counter+1}/{total_batches}], Loss: {loss.item():.4f}, Progress: {progress:.2f}%")
            batch_counter += 1
            
        # Print progress
        if TRAINING_PRINT_EVERY_EPOCHS >= 0 and (epoch + 1) % TRAINING_PRINT_EVERY_EPOCHS == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def evaluateModel(model, RNAcompete_sequences_path, RNCMPT_training_path, batch_size=256, output_path = 'output_file.csv'):
    print('Start evaluate!')
    # Create the custom dataset
    eval_dataset = RNCMPTDataset(RNAcompete_sequences_path, RNCMPT_training_path)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Set the model to evaluation mode
    model.eval()

    # Evaluation loop
    predicts, truths = [], []
    with torch.no_grad():
        batch_count = 0
        total_batchs = len(eval_loader)
        for rna_sequence, binding_score in eval_loader:
            
            rna_sequence = rna_sequence.to(device)
            binding_score = binding_score.tolist()
            for truth in binding_score:
                truths.append(truth[0])
            # Pass the input through the model to get the predicted score
            predicted_score = model(rna_sequence)
            predicted_score = predicted_score.tolist()
            for predicted in predicted_score:
                predicts.append(predicted[0])

            # Calculate progress and print
            progress = batch_count / total_batchs * 100
            # Print progress
            if EVALUATION_PRINT_EVERY_BATCHES >= 0 and (batch_count + 1) % EVALUATION_PRINT_EVERY_BATCHES == 0:
                print(f"Batch [{batch_count+1}/{total_batchs}], Progress: {progress:.2f}%")
            batch_count += 1

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
    # TODO: not sure if this is how we suppose to compare to the real outputs.
    print('Pearson correlation:', pearson_correlation.item())
    return pearson_correlation.item()

def processOnce(rbns_file_paths, RNAcompete_sequences_path, RNCMPT_training_path, RBP_num):
    start_time = time.time()
    
    # seqs_per_file, batch_size, epochs, learning_rate = 50000, 32, 3, 0.001
    seqs_per_file, batch_size, epochs, learning_rate = 16000, 32, 15, 0.002
    print('Run with Hyperparams:\n', 'Seqs per file:', seqs_per_file, ', Batch size:', batch_size, ', Epochs:', epochs, ', Learning Rate:', learning_rate)
    model = createModel()
    trainModel(model, rbns_file_paths, seqs_per_file, batch_size, epochs, learning_rate)
    pearson_correlation = evaluateModel(model, RNAcompete_sequences_path, RNCMPT_training_path,output_path=f'outputs/RBP{RBP_num}_output.csv')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

# ----
# TODO: everything from here and forward is probably should move from here (This script should be only the core logic)

def generateHyperparams():
    # Define the lists for each hyperparameter
    seqs_per_file_list = [16000, 32000, 64000, 128000]
    batch_size_list = [16, 32, 64, 128, 256, 512]
    epochs_list = [3, 5, 7, 10]
    learning_rate_list = [0.00001 ,0.0001, 0.001, 0.01]

    # Generate random hyperparameters by randomly selecting from the lists
    seqs_per_file = random.choice(seqs_per_file_list)
    batch_size = random.choice(batch_size_list)
    epochs = random.choice(epochs_list)
    learning_rate = random.choice(learning_rate_list)

    return seqs_per_file, batch_size, epochs, learning_rate

def serachParams(rbns_file_paths, RNAcompete_sequences_path, RNCMPT_training_path, csv_filename = "results.csv"):
    # Open the CSV file in write mode and create a CSV writer object
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row to the CSV file
        writer.writerow(["Seqs per file", "Batch size", "Epochs", "Learning Rate", "Pearson Correlation", "Elapsed Time"])
        # Loop for 20 iterations and write results for each iteration
        for i in range(20):
            start_time = time.time()
            seqs_per_file, batch_size, epochs, learning_rate = generateHyperparams()
            print('Testing Hyperparams:\n', 'Seqs per file:', seqs_per_file, ', Batch size:', batch_size, ', Epochs:', epochs, ', Learning Rate:', learning_rate)            
            model = createModel()
            trainModel(model, rbns_file_paths, seqs_per_file, batch_size, epochs, learning_rate)
            pearson_correlation = evaluateModel(model, RNAcompete_sequences_path, RNCMPT_training_path, output_path=f'outputs/search_seed{torch.seed()}_seqs{seqs_per_file}_batchSize{batch_size}_epochs{epochs}_lr{learning_rate}.csv')
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed Time: {elapsed_time} seconds")

            # Write the results to the CSV file
            writer.writerow([seqs_per_file, batch_size, epochs, learning_rate, pearson_correlation, elapsed_time])


if __name__ == '__main__':
    rbns_file_paths = [
                       'RBNS_training/RBP1_input.seq',
                       'RBNS_training/RBP1_5nM.seq',
                       'RBNS_training/RBP1_20nM.seq',
                       'RBNS_training/RBP1_80nM.seq',
                       'RBNS_training/RBP1_320nM.seq',
                       'RBNS_training/RBP1_1300nM.seq'
                       ]
    RNAcompete_sequences_path = "RNAcompete_sequences.txt"
    RNCMPT_training_path = "RNCMPT_training/RBP1.txt"
    # serachParams(rbns_file_paths, RNAcompete_sequences_path, RNCMPT_training_path)
    processOnce(rbns_file_paths, RNAcompete_sequences_path, RNCMPT_training_path, 1)