import os
import time
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , Dataset , SubsetRandomSampler
from torchsummary import summary
import pandas as pd
import numpy as np
import itertools

# Set the seed for PyTorch
SEED = random.randrange(0,123456789)
print(SEED)
torch.manual_seed(SEED)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepSELEXModel(nn.Module):
    def __init__(self, classes = 6):
        super(DeepSELEXModel, self).__init__()
        kernelsPerConv = 512
        inputSize = 20
        poolKernelStride = 5
        # First Conv1d layer with kernel size 3
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=8, stride=1, padding=4)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=poolKernelStride, stride=poolKernelStride)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(kernelsPerConv * (inputSize // 5), 64),  # input size = 2048 is the number of output channels from each Conv1d layer
            nn.ReLU(),
            nn.Dropout(0.65),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.65),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.65),
            nn.Linear(32, classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)

        # Apply the three Conv1d layers with different kernel sizes
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Reshape x for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)
        return x

def one_hot_encoding(rna_sequence, nucleotides = ['A', 'C', 'G', 'T'], max_length=20, padding_value=0.25):
    one_hot_sequence = []
    for nucleotide in rna_sequence:
        encoding = [int(nucleotide == nuc) for nuc in nucleotides]
        one_hot_sequence.append(encoding)

    # Calculate the amount of padding required on each side
    total_padding = max_length - len(one_hot_sequence)
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    # Append padding rows with the desired value on both sides
    if left_padding > 0:
        padding_row = [padding_value] * len(nucleotides)
        padding_rows = [padding_row] * left_padding
        one_hot_sequence = padding_rows + one_hot_sequence

    if right_padding > 0:
        padding_row = [padding_value] * len(nucleotides)
        padding_rows = [padding_row] * right_padding
        one_hot_sequence = one_hot_sequence + padding_rows

    return np.array(one_hot_sequence)


# Custom Dataset to read and process data in chunks
class RBNSDataset(Dataset):
    def __init__(self, file_paths, nrows_per_file = -1):
        self.nrows_per_file = nrows_per_file
        self.file_paths = file_paths
        self.file_seqs, self.file_targets, self.cumulative_lengths = self._load_data()
    def __len__(self):
        return len(self.file_seqs)
    
    def _load_data(self):
        # Load and concatenate data from multiple files
        grad_arr = [index/len(self.file_paths) for index in range(len(self.file_paths))]
        data_list = []
        target_list = []
        cumulative_lengths = [0]  # Store the cumulative lengths of the files
        for index in range(len(self.file_paths)):
            file_path = self.file_paths[index]
            file_data = pd.read_csv(file_path, delimiter='\t', header=None)
            if self.nrows_per_file >= 0:
                file_data = file_data.sample(n=self.nrows_per_file, random_state=SEED)
            data_list.append(file_data)
            onehot = one_hot_encoding([index/len(self.file_paths)],grad_arr, 1, 0)[0]
            target_list.extend([onehot] * len(file_data))
            cumulative_lengths.append(cumulative_lengths[-1] + len(file_data))
            print('Loaded seqs file: ', file_path)
        return pd.concat(data_list, ignore_index=True), target_list, cumulative_lengths

    def _get_file_index(self,index):
        file_index = sum(1 for length in self.cumulative_lengths if length <= index)
        return file_index
    
    def getClassesNum(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.file_seqs.iloc[index, 0]
        binding_score = self.file_targets[index]

        # Preprocess the RNA sequence (if needed)
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence)

        # Convert RNA sequence to tensor
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)

        # Set the target variable.
        target = torch.tensor(binding_score, dtype=torch.float32)

        return input_data, target

# Custom Dataset to read and process data in chunks
class RNCMPTDataset(Dataset):
    def __init__(self, RNAcompete_sequences_path, RNCMPT_training_path):
        with open(RNAcompete_sequences_path, 'r') as f:
            self.seqs_data = f.readlines()
        
        with open(RNCMPT_training_path, 'r') as f:
            self.targets_data = [float(line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.seqs_data)

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.seqs_data[index].strip()
        binding_score = torch.tensor(self.targets_data[index], dtype=torch.float32)
        
        # Preprocess the RNA sequence (if needed)
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence, ['A', 'C', 'G', 'U'])

        # Convert RNA sequence to tensor
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)

        # Set the target variable.
        target = torch.tensor(np.array([binding_score]), dtype=torch.float32)

        return input_data, target

def createModel(classesNum):
    model = DeepSELEXModel(classesNum).to(device) # NOTE: Set the model we want to run here.
    # summary(model, (42, 4))
    return model

def trainModel(rbns_file_paths):
    # hyperparams:
    seqs_per_file = 15000
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    weight_decay= 0.00001
    betas=(0.9,0.999)
    test_ratio=0.3
    patience=3
    print(f'Run with Hyperparams:\nNumber of Seqs: {seqs_per_file}, Batch_size: {batch_size}, Number of Epochs: {num_epochs},\nLearning Rate: {learning_rate}, Weight Decay: {weight_decay}, Betas: {betas}, Test Ratio: {test_ratio}, Patience: {patience}')
    print('Load Data!')
    rbns_dataset = RBNSDataset(rbns_file_paths,seqs_per_file)

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
    

    model = createModel(rbns_dataset.getClassesNum())

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay= weight_decay) # NOTE: Adam used originally but I find it getting stuck really quickly on local minimal (making the model useless). SGD seems to run better.
            # optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay, betas=betas)
    
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
    
    return model

def evaluateModel(model, RNAcompete_sequences_path, RNCMPT_training_path, output_path = 'output_file.csv'):
    print(f'Start evaluate! Combination={(-1,3,4)}')
    # Create the custom dataset
    eval_dataset = RNCMPTDataset(RNAcompete_sequences_path, RNCMPT_training_path)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Set the model to evaluation mode
    model.eval()
    classesNum = 6 # TODO: HARDCODED, send it to function from training.
    # Evaluation loop
    predicts, truths = [], []
    with torch.no_grad():
        total = len(eval_loader)
        count = 0
        for sequences, targets in eval_loader:
            count += 1
            sequences = sequences.to(device)
            sequence = sequences[0]
                
            # Run sequences in a sliding window, sized 20 (as in the training set)
            window_size = 20
            stride = 1
            predicted_columns = np.empty((0, classesNum))

            # Collect all windows in a list
            windows = [sequence[i : i + window_size] for i in range(0, len(sequence) - window_size + 1, stride)]

            # Convert the list of windows into a single tensor
            windows_tensor = torch.stack(windows)

            # Pass the input through the model to get the predicted scores for all windows
            predicted_matrices = model(windows_tensor)

            # Convert the predictions to a numpy array
            predicted_columns = predicted_matrices.cpu().numpy()

            # Reshape the predictions to get predicted_scores for each window
            predicted_columns = predicted_columns.reshape(-1, classesNum)

            column_0 = predicted_columns[:, 0]
            column_3 = predicted_columns[:, 3]
            column_4 = predicted_columns[:, 4]
            # Calculate the predicted_score using the formula
            predicted_score = -np.min(column_0) + np.max(column_3) + np.max(column_4)
            predicts.append(predicted_score)

            # Save targets.
            targets = targets.tolist()
            for target in targets:
                truths.append(target[0]) 

            if count % 1000 == 0: 
                print(predicted_score, count, total)

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

# Function to calculate the predicted score for a given combination of elements
def calculate_predicted_score(predicted_vector, combination):
    # `combination` is a tuple containing the indices of the elements in the predicted_vector to use
    return -predicted_vector[0] + predicted_vector[3] + predicted_vector[4]
    #return sum(predicted_vector[idx] for idx in combination)

def backup_single_run():
    rbp_num = 9
    rbns_file_paths = [
                       f'RBNS_training/RBP{rbp_num}_input.seq',
                       f'RBNS_training/RBP{rbp_num}_5nM.seq',
                       f'RBNS_training/RBP{rbp_num}_20nM.seq',
                       f'RBNS_training/RBP{rbp_num}_80nM.seq',
                       f'RBNS_training/RBP{rbp_num}_320nM.seq',
                       f'RBNS_training/RBP{rbp_num}_1300nM.seq'
                       ]
    
    RNAcompete_sequences_path = "RNAcompete_sequences.txt"
    RNCMPT_training_path = f"RNCMPT_training/RBP{rbp_num}.txt"

    start_time = time.time()

    model = trainModel(rbns_file_paths)

    # Generate all combinations of indices from 0 to 5 (since there are 6 elements in the vector)
    all_combinations = itertools.combinations(range(6), 3)
    # Loop over all combinations and calculate the predicted score
    for combination in all_combinations:
        pearson_correlation = evaluateModel(model, RNAcompete_sequences_path, RNCMPT_training_path,combination,output_path=f'outputs/RBP{rbp_num}_output.csv')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

if __name__ == '__main__':
    # TODO: it crashes because it isn't the exact files for each RBPs, should do something like scaning the folder for RBPX_{}.seq files, and send those sorted to the function.
    #       Look at the RBP11.
    num_rbns_files = 10
    # Loop through the RBPs and process each one
    for rbp_num in range(1, num_rbns_files + 1):
        rbns_file_paths_rbp = [
            f'RBNS_training/RBP{rbp_num}_input.seq',
            f'RBNS_training/RBP{rbp_num}_5nM.seq',
            f'RBNS_training/RBP{rbp_num}_20nM.seq',
            f'RBNS_training/RBP{rbp_num}_80nM.seq',
            f'RBNS_training/RBP{rbp_num}_320nM.seq',
            f'RBNS_training/RBP{rbp_num}_1300nM.seq'
        ]
        RNAcompete_sequences_path_rbp = f"RNAcompete_sequences.txt"
        RNCMPT_training_path_rbp = f"RNCMPT_training/RBP{rbp_num}.txt"

        start_time = time.time()

        model = trainModel(rbns_file_paths_rbp)

        pearson_correlation = evaluateModel(model, RNAcompete_sequences_path_rbp, RNCMPT_training_path_rbp,output_path=f'outputs/RBP{rbp_num}_output.csv')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")