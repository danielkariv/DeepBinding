import os
import sys
import time
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader , SubsetRandomSampler, Dataset
from torchsummary import summary
import pandas as pd
import numpy as np

# Custom Dataset to read and process data in chunks
class RBNSDataset(Dataset):
    def __init__(self, RBNS_paths, nrows_per_file = -1):
        self.RBNS_paths = RBNS_paths
        self.nrows_per_file = nrows_per_file
        self.file_seqs, self.cumulative_lengths = self._load_data()
        # After loading, we want to save what the input shape and classes should be.
        self.classesNum = len(self.RBNS_paths)
        seq,target = self.__getitem__(0) # NOTE: I don't like this, but it is the fastest way to get the shape.
        self.inputShape = seq.shape
        print(self.classesNum, self.inputShape)
        print('Seqs loaded: ', len(self.file_seqs))
    def __len__(self):
        return len(self.file_seqs)
    
    # NOTE: It loads the top sequences in the files, with one-hot for input and output.
    def _load_data(self):
        # Load and concatenate data from multiple files
        data_list = []
        cumulative_lengths = [0]  # Store the cumulative lengths of the files
        # run over all the files and load them.
        for index in range(len(self.RBNS_paths)):
            # Get path to file.
            file_path = self.RBNS_paths[index]
            # Load it, and getting a sample of it if we limited the amount.
            file_data = pd.read_csv(file_path, delimiter='\t', header=None, usecols=[0,1])
            file_data.columns = ['RNA', 'Counts']
            file_data = file_data.sort_values(by='Counts', ascending=False)

            if self.nrows_per_file >= 0:
                # Select the most frequent sequences up to nrows_per_file
                file_data = file_data.head(self.nrows_per_file)
            # Adds sequences. 
            data_list.extend(file_data['RNA'])
            seqs_count = len(file_data['RNA'])
            
            cumulative_lengths.append(cumulative_lengths[-1] + seqs_count)
            print('Loaded seqs file: ', file_path)
        
        return data_list, cumulative_lengths
    
    def _get_file_index(self,index):
        file_index = sum(1 for length in self.cumulative_lengths if length <= index)
        return file_index
    
    def getClassesNum(self):
        return self.classesNum
    
    def getInputShape(self):
        return self.inputShape

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.file_seqs[index]
        # preprocess the RNA Sequence by creating an one hot encoding, with padding and max length (RBNS has sequences with the same size, but in RNCMPT they are different in size and so is needed)
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence,['A','C','G','T'],max_length=len(rna_sequence),padding_value=0.25)

        # Create a one-hot representing of the class (the file it comes from)
        file_index = self._get_file_index(index)
        target = [file_index]
        classes = [index for index in range(self.classesNum)]
        target_onehot = one_hot_encoding(target, classes, 1, 0)[0]

        # Convert the input/ouput to tensors for usage in pytorch.
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)
        target = torch.tensor(target_onehot, dtype=torch.float32)

        return input_data, target

# Custom Dataset to read and process data in chunks
class RNCMPTDataset(Dataset):
    def __init__(self, RNAcompete_sequences_path, expectedSeqLength):
        with open(RNAcompete_sequences_path, 'r') as f:
            self.seqs_data = f.readlines()
        # NOTE: because training set has different sizes depending on the RBP, we recieve the expected Sequence length, and use it to pad the sequences that we want to run on.
        self.expectedSeqLength = expectedSeqLength
    def __len__(self):
        return len(self.seqs_data)

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.seqs_data[index].strip()
        binding_score = torch.tensor(self.targets_data[index], dtype=torch.float32)
        
        # Preprocess the RNA Sequence by creating an one hot encoding, with padding and max length
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence, ['A', 'C', 'G', 'U'], max_length=self.expectedSeqLength, padding_value=0.25)
        
        # Convert the input/ouput to tensors for usage in pytorch.
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)
        target = torch.tensor(np.array([binding_score]), dtype=torch.float32)

        return input_data, target

class TransformerModel(nn.Module):
    def __init__(self, inputShape=(20, 4), classes=6, d_model=32, nhead=4, num_encoder_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(inputShape[1], d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.fc1 = nn.Linear(inputShape[0] * d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Reshape for transformer input (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_len, d_model)
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output_layer(x)
        
        return x

def receiveArgs():
    args_count = len(sys.argv) - 1  # Ignore the 1st, it is the script name.
    if args_count >= 6 and args_count <= 8: # includes: ofile, RNCMPT, input, RBNS1, RBNS2, .. RBNS5
        ofile = sys.argv[1]
        RNCMPT = sys.argv[2]
        RBNS = sys.argv[3:] # Should be sorted by how it given(for example: input, 5nm, ... ,3200nm)
        print(ofile, RNCMPT, RBNS)
        return ofile, RNCMPT, RBNS
    else:
        print("No enough arguments provided.")
        sys.exit(1) # Exit with an error status

def loadTrainTestLoaders(rbns_file_paths, seqs_per_file = 1000, batch_size = 128, test_ratio = 0.3):
    rbns_dataset = RBNSDataset(rbns_file_paths, seqs_per_file)

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

def train(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}")
        scheduler.step()

def predict(ofile, RNCMPT, inputShape):
    eval_dataset = RNCMPTDataset(RNCMPT, inputShape[0])
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False) # TODO: support multiple sequences at once, batch_size > 1 (GPU can support with best speed at 512 or 1024, could run in seconds instead of few minutes)
    # Set the model to evaluation mode
    model.eval()
    predicts = []
    with torch.no_grad():
        total = len(eval_loader)
        count = 0
        for sequences, targets in eval_loader:
            count += len(sequences)
            sequences = sequences.to(device)
            # sequence = sequences[0] # TODO: support multiple sequences at once, batch_size > 1
            
            # Pass the input through the model to get the predicted scores for all windows
            predicted_matrices = model(sequences)

            # Convert the predictions to a numpy array
            predicted_columns = predicted_matrices.cpu().numpy()

            # Reshape the predictions to get predicted_scores for each window
            predicted_columns = predicted_columns.reshape(-1, classesNum)
            # Calculate the score for each sequence
            sequence_scores = []
            for row in predicted_columns:
                print(row)
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
    with open(ofile, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Convert the predicts list to a list of lists
        predicts_list = [[value] for value in predicts]
        # Write each list as a row in the CSV file
        for row in predicts_list:
            writer.writerow(row)
    return predicts

def compare(predicts, targets):
    # Convert the lists to tensors
    epsilon = 1e-9  # A small value to avoid division by zero
    predicts_tensor = torch.tensor(predicts) + epsilon
    targets_tensor = torch.tensor(targets) + epsilon

    combined_tensor = torch.stack((predicts_tensor, targets_tensor), dim=0)

    # Calculate Pearson correlation
    correlation_matrix = torch.corrcoef(combined_tensor)
    pearson_correlation = correlation_matrix[0, 1]
    
    return pearson_correlation

if __name__ == '__main__':
    ### Hyperparams:
    learning_rate = 1e-3
    warmup_epochs = 5
    num_epochs = 10
    ofile, RNCMPT, RBNS = receiveArgs()
    ### Initilize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = loadTrainTestLoaders(RBNS) # TODO: needs more cleanup.
    inputShape, classesNum = train_loader.dataset.getInputShape(),train_loader.dataset.getClassesNum()
    
    model = TransformerModel(inputShape,classesNum).to(device) # NOTE: Set the model we want to run here.

    ###  Training process:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs))

    train(model, train_loader, criterion, optimizer, scheduler, num_epochs)

    predicts = predict(ofile, RNCMPT, inputShape)


    # Isn't part of the final code.
    ### Loads the training data and compare with Pearson
    RNCMPT_training_path = "datasets\RNCMPT_training\RBP1.txt"
    with open(RNCMPT_training_path, 'r') as f:
        targets_data = [float(line.strip()) for line in f.readlines()]
        pearson = compare(predicts, targets_data)
        print(pearson)
    

