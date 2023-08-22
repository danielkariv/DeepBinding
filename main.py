import sys
import os
import time
import csv
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader , SubsetRandomSampler
# from torchsummary import summary
import pandas as pd
import numpy as np

# Set the seed for PyTorch
SEED = 93015700 # random.randrange(0,123456789)
print('Seed: ', SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert sequence to one-hot encoding, with padding if needed.
def one_hot_encoding(rna_sequence, nucleotides = ['A', 'C', 'G', 'T'], max_length=20, padding_value=0.25):
    one_hot_sequence = []
    for nucleotide in rna_sequence:
        if nucleotide == 'N':
            encoding = [padding_value] * len(nucleotides)
        else:
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

# Custom Dataset to read and process data in chunks of RBNS files.
class RBNSDataset(Dataset):
    def __init__(self, file_paths, SEED, nrows_per_file = -1):
        self.file_paths = file_paths
        self.SEED = SEED
        self.nrows_per_file = nrows_per_file
        self.seq_length = 41 # =Max
        self.file_seqs, self.cumulative_lengths = self._load_data()
        print('Seqs loaded: ', len(self.file_seqs))
    def __len__(self):
        return len(self.file_seqs)
    
    def _load_data(self):
        # Load and concatenate data from multiple files
        data_list = []
        cumulative_lengths = [0]  # Store the cumulative lengths of the files
        # run over all the files and load them.
        for index in range(len(self.file_paths)):
            # Get path to file.
            file_path = self.file_paths[index]
            # Load it, and getting a sample of it if we limited the amount.
            file_data = pd.read_csv(file_path, delimiter='\t', header=None, usecols=[0,1])
            file_data.columns = ['RNA', 'Counts']
            file_data = file_data.sort_values(by='Counts', ascending=False)

            if self.nrows_per_file >= 0:
                # Select the most frequent sequences up to nrows_per_file
                file_data = file_data.head(self.nrows_per_file)

             # Decide if to take one one per row, or based on the count, duplicate it.
            is_duplicate_seqs = False
            # Loads top seqeunces, duplicate them based on the count.
            if is_duplicate_seqs: 
                total_added = 0
                # Iterate through each row and add RNA sequences based on their counts
                for _, row in file_data.iterrows():
                    sequence = row['RNA']
                    count = row['Counts']
                    # Add the sequence 'count' number of times to the data_list
                    data_list.extend([sequence] * count)
                    total_added += count

                    if total_added >= self.nrows_per_file:
                        break
                
                cumulative_lengths.append(cumulative_lengths[-1] + total_added)
            # load the top sequences, ignoring the count.
            else: 
                data_list.extend(file_data['RNA'])
                count = len(file_data['RNA'])

                cumulative_lengths.append(cumulative_lengths[-1] + count)

            print('Loaded seqs file: ', file_path)
        
        return data_list, cumulative_lengths

    def _get_file_index(self,index):
        file_index = sum(1 for length in self.cumulative_lengths if length <= index)
        return file_index
    
    def getClassesNum(self):
        return len(self.file_paths)
    
    def getInputShape(self):
        input_data, _ = self.__getitem__(0)
        return input_data.shape

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.file_seqs[index]
        
        # preprocess the RNA Sequence by creating an one hot encoding, with padding and max length (RBNS has sequences with the same size, but in RNCMPT they are different in size and so is needed)
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence,['A','C','G','T'],max_length=self.seq_length ,padding_value=0.25)

        # Create a one-hot representing of the class (the file it comes from)
        file_index = self._get_file_index(index)
        target = [file_index]
        classes = [index for index in range(len(self.file_paths))]
        target_onehot = one_hot_encoding(target, classes, 1, 0)[0]

        # Convert the input/ouput to tensors for usage in pytorch.
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)
        target = torch.tensor(target_onehot, dtype=torch.float32)
        
        return input_data, target

# Custom Dataset to read and process data in chunks of RNCMPT file.
class RNCMPTDataset(Dataset):
    def __init__(self, RNAcompete_sequences_path, modelExpectedSeqLength):
        with open(RNAcompete_sequences_path, 'r') as f:
            self.seqs_data = f.readlines()
        # NOTE: because training set has different sizes depending on the RBP, we recieve the expected Sequence length, and use it to pad the sequences that we want to run on.
        self.modelExpectedSeqLength = modelExpectedSeqLength
    def __len__(self):
        return len(self.seqs_data)

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.seqs_data[index].strip()
        
        # Preprocess the RNA Sequence by creating an one hot encoding, with padding and max length
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence, ['A', 'C', 'G', 'U'], max_length=self.modelExpectedSeqLength, padding_value=0.25)
        
        # Convert the input/ouput to tensors for usage in pytorch.
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)

        return input_data

# Model is taking only max value for each kernel, and then running on them with FC layers.
# NOTE: Correlation: 0.162162392025289 (file: pearson_2023-08-21_03-29-37.csv)
class DeepConvModel4(nn.Module):
    def __init__(self, inputShape = (20,4), classes = 6, hidden_size = 128, conv_chs = 512, kernel_size = 4) :
        super(DeepConvModel4, self).__init__()
        # First Conv1D layer
        self.conv_layer1 = nn.Conv1d(in_channels=inputShape[1], out_channels=conv_chs, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.flatten = nn.Flatten()
        # Define the layers
        input_size = conv_chs
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)
        # Conv1D layer 1
        x = self.conv_layer1(x)
        x, _ = torch.max(x, dim=2)  # along the time dimension
        x = self.relu(x)

        # Flatten the tensor
        x = self.flatten(x)

        # Fully connected layers
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))

        # Output layer
        x = self.output_layer(x)
        x = self.softmax(x)

        return x
# Create the model.
def createModel(inputShape, classesNum):
    model = DeepConvModel4(inputShape,classesNum).to(device) 
    # summary(model, inputShape) # NOTE: used to debug the model design.
    return model

# Create Train/Test loaders.
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

# Pipeline used to Train/create a model and then to predict on it.
class NormalPipeline:
    def __init__(self):
        print('Run with normal pipeline.')
        
    def trainModel(self, rbns_file_paths):
        # Hyperparams:
        seqs_per_file = 15000
        batch_size = 64
        num_epochs = 30
        learning_rate = 1e-3
        weight_decay= 1e-5
        betas=(0.9,0.999)
        test_ratio= 0.1 # 0.3
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
    
    def predict(self, RNAcompete_sequences_path):
        print(f'Start evaluate!')
        # Create the custom dataset
        eval_dataset = RNCMPTDataset(RNAcompete_sequences_path, self.inputShape[0])
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False) # TODO: support multiple sequences at once, batch_size > 1 (GPU can support with best speed at 512 or 1024, could run in seconds instead of few minutes)

        # Set the model to evaluation mode
        self.model.eval()
        # Evaluation loop
        outputs = []
        with torch.no_grad():
            total = len(eval_loader)
            count = 0
            for sequences in eval_loader:
                count += len(sequences)
                sequences = sequences.to(device)

                # Pass the entire batch of sequences through the model
                predicted_matrices = self.model(sequences)

                # Convert the predictions to a numpy array
                predicted_columns = predicted_matrices.cpu().numpy()

                # Calculate the predicted_scores for each item in the batch
                batch_outputs = []
                for i in range(len(sequences)):
                    predicted_output = predicted_columns[i]
                    batch_outputs.append(predicted_output)

                outputs.extend(batch_outputs)

        return outputs

# Takes predicts/truths and calculate pearson between the two sets.
# Used for debug feature.
def pearson_compare(predicts, truths):
    # Convert the lists to tensors
    predicts_tensor = torch.tensor(predicts)
    truths_tensor = torch.tensor(truths)

    combined_tensor = torch.stack((predicts_tensor, truths_tensor), dim=0)

    # Calculate Pearson correlation
    correlation_matrix = torch.corrcoef(combined_tensor)
    pearson_correlation = correlation_matrix[0, 1]
    # return the value.
    return pearson_correlation.item()

# Helper function to get all the RBP files, as a list.
# Used for debug feature.
def find_rbp_files(dir_path, rbp_num):
    rbp_files = [
        f'{dir_path}/{file}'
        for file in os.listdir(f'{dir_path}')
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

# Gets the args as varibles.
def receiveArgs():
    args_count = len(sys.argv) - 1  # Ignore the 1st, it is the script name.
    if args_count >= 6 and args_count <= 8: # includes: ofile, RNCMPT, input, RBNS1, RBNS2, .. RBNS5
        ofile = sys.argv[1]
        RNCMPT = sys.argv[2]
        RBNS = sys.argv[3:] # Should be sorted by how it given(for example: input, 5nm, ... ,3200nm)
        print(ofile, RNCMPT, RBNS)
        return ofile, RNCMPT, RBNS
    else:
        print("No correct amount of arguments was provided (allows output-file, RNCMPT file, and between 4-6 RBNS files).")
        sys.exit(1) # Exit with an error status

# Function that process given args, including train and predict process
def procressLogic(ofile, RNCMPT, RBNS):
    # Initilize pipeline.
    sp = NormalPipeline()
    # Train the model.
    start_time = time.time()
    sp.trainModel(RBNS)
    end_time = time.time()
    elapsed_time = end_time - start_time
    train_time = elapsed_time
    print(f"Elapsed Time for training: {elapsed_time} seconds")
    # Get Predictions pre-processed.
    start_time = time.time()
    predicted_outputs = sp.predict(RNCMPT)
    end_time = time.time()
    elapsed_time = end_time - start_time
    evalute_time = elapsed_time
    print(f"Elapsed Time for predicts: {elapsed_time} seconds")
    # Process outputs to binding scores.
    
    # Get length of expected outputs that got predicted
    num_outputs = len(predicted_outputs[0])
    predictions = []
    for predict_output in predicted_outputs:
        # TODO: find a good function for scoring the binding.
        # NOTE: for now, this is seems like the most balance score function. Some RBPs, loves one file over the other, but the higher ones usally the positive, and the lower ones are usally the negative.
        score = - sum(predict_output[:num_outputs // 2]) + sum(predict_output[num_outputs // 2 + 1:])
        predictions.append(score)

    # Open the file in write mode
    with open(ofile, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Convert the predicts list to a list of lists
        predicts_list = [[value] for value in predictions]
        # Write each list as a row in the CSV file
        for row in predicts_list:
            writer.writerow(row)
    
    return predictions, train_time, evalute_time

if __name__ == '__main__':
    # NOTE: Commend like this need to be written (basically as we instructed):
    #      'python main.py test.txt datasets\RNAcompete_sequences.txt datasets\RBNS_training\RBP1_input.seq datasets\RBNS_training\RBP1_5nm.seq datasets\RBNS_training\RBP1_20nm.seq datasets\RBNS_training\RBP1_80nm.seq datasets\RBNS_training\RBP1_320nm.seq datasets\RBNS_training\RBP1_1300nm.seq'
    DebugFlag = False
    CompareFlag = False
    if DebugFlag == False:
        # Recieve paths from args.
        ofile, RNCMPT, RBNS = receiveArgs()
        # Run training on model, and predict the results.
        predictions = procressLogic(ofile, RNCMPT, RBNS)
    else:
        # NOTE: A debug feature that allows testing all 16 RBPs in the training set or train over all testing set.
        # It will run on each, all the files, and report back times, and pearson corrrelation per RBP if allowed to do so.
        rbp_pearson_list = []
        for rbp_num in range(1, 16 + 1):
            # Create paths to use with.
            dir_path = 'D:/DeepBinding/datasets/' # '.' # HARDCODED path.
            rbns_file_paths_rbp = find_rbp_files(dir_path + "RBNS_testing/", rbp_num) # Automatically find all the files and send them sorted in the right way.
            RNAcompete_sequences_path_rbp = f"{dir_path}/RNAcompete_sequences.txt"
            ofile, RNCMPT, RBNS = f'outputs/RBP{rbp_num}_output.csv', RNAcompete_sequences_path_rbp, rbns_file_paths_rbp
        
            # Run training on model, and predict the results.
            predictions, train_time, evalute_time = procressLogic(ofile, RNCMPT, RBNS)
            if CompareFlag:
                # Load real data and compare with the predicted.
                RNCMPT_training_path_rbp = f"{dir_path}/RNCMPT_training/RBP{rbp_num}.txt"
                with open(RNCMPT_training_path_rbp, 'r') as f:
                    targets = [float(line.strip()) for line in f.readlines()]
                pearson_correlation = pearson_compare(predictions, targets)
                print(f'RBP {rbp_num}: Pearson correlation =', pearson_correlation)
                rbp_pearson_list.append((f'RBP {rbp_num}', pearson_correlation, train_time, evalute_time))

        if CompareFlag:
            # Get the current date and time
            current_datetime = datetime.datetime.now()
            # Convert the datetime object to a string
            current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

            # Save debug results to a file.
            with open(f'pearson_{current_datetime_str}.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['RBP', 'Pearson Correlation', 'Train Time (Seconds)', 'Evalute Time (Seconds)'])
                sum_rbps = 0
                for item in rbp_pearson_list:
                    sum_rbps += item[1]
                    writer.writerow([item[0], item[1], item[2], item[3]])
                sum_rbps = sum_rbps / len(rbp_pearson_list)
                print('Average Correlation: ', sum_rbps)
                writer.writerow(['Average', sum_rbps, '',''])

        print('Finished running!')
    
