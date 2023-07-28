import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.Datasets import RBNSDataset, RNCMPTDataset
from torchsummary import summary
import time
import csv
import random
# Set the seed for PyTorch
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple neural network
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        # NOTE: based on DeepSELEX (without the output cycles and sliding window).
        #       Performance aren't good, it finds correlation of 0.05-0.15.
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=8, stride=1)
        self.maxpool_layer = nn.MaxPool1d(kernel_size=5, stride=5)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7, 64), # NOTE: input is based on size of maxpool (512,7).
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Sigmoid(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)

        # Apply convolution layer
        x = self.conv_layer(x)

        # Apply max pooling
        x = self.maxpool_layer(x)

        # Reshape x for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)
        return x
def createModel():
    model = NNModel().to(device)
    summary(model, (42, 4))
    return model

def trainModel(model, seqs_per_file, batch_size, epochs, learning_rate):
    print('Load Data!')
    # NOTE: It loads to memory the seqs. it takes around 20gb on RAM, so run it on a system that can process it.
    #       We could process it by loading when needed a row from it but it is really slow.
    rbns_file_paths = ['RBNS_training/RBP1_input.seq','RBNS_training/RBP1_5nM.seq','RBNS_training/RBP1_20nM.seq','RBNS_training/RBP1_80nM.seq','RBNS_training/RBP1_320nM.seq','RBNS_training/RBP1_1300nM.seq']
    rbns_dataset = RBNSDataset(rbns_file_paths,seqs_per_file) 
    data_loader = DataLoader(rbns_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    num_epochs = epochs
    print_every_epochs = 1
    print_every_batchs = 200
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Start Training!')
    # Training loop
    total_batches = len(data_loader)
    for epoch in range(num_epochs):
        batch_counter = 0  # Counter to track the current batch index within the epoch

        for batch_data, targets in data_loader:
            batch_counter += 1
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
            if (batch_counter + 1) % print_every_batchs == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_counter}/{total_batches}], Loss: {loss.item():.4f}, Progress: {progress:.2f}%")

            
        # Print progress
        if (epoch + 1) % print_every_epochs == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def evaluateModel(model):
    print('Start evaluate!')
    # Evaluate
    RNAcompete_sequences_path = "RNAcompete_sequences.txt"
    RNCMPT_training_path = "RNCMPT_training/RBP1.txt"
    # Create the custom dataset
    eval_dataset = RNCMPTDataset(RNAcompete_sequences_path, RNCMPT_training_path)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    # Set the model to evaluation mode
    model.eval()

    # Evaluation loop
    print_every_batchs = 10
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
            if (batch_count + 1) % print_every_batchs == 0:
                print(f"Batch [{batch_count+1}/{total_batchs}], Progress: {progress:.2f}%")
            batch_count+= 1

    # Specify the file path
    file_path = 'output_file.csv'

    # Open the file in write mode
    with open(file_path, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Convert the predicts list to a list of lists
        predicts_list = [[value] for value in predicts]
        # Write each list as a row in the CSV file
        for row in predicts_list:
            writer.writerow(row)

    # Convert the lists to tensors
    predicts_tensor = torch.tensor(predicts)
    truths_tensor = torch.tensor(truths)

    combined_tensor = torch.stack((predicts_tensor, truths_tensor), dim=0)

    # Calculate Pearson correlation
    correlation_matrix = torch.corrcoef(combined_tensor)
    pearson_correlation = correlation_matrix[0, 1]
    print('Pearson correlation:', pearson_correlation.item())
    return pearson_correlation.item()

def generateHyperparams():
    # Define the range for each hyperparameter
    seqs_per_file_range = (16000, 64000)
    batch_size_range = (16, 64)
    epochs_range = (3, 10)
    learning_rate_range = (0.0001, 0.01)

    # Generate random hyperparameters within the specified ranges
    seqs_per_file = random.randint(*seqs_per_file_range)
    batch_size = random.randint(*batch_size_range)
    epochs = random.randint(*epochs_range)
    learning_rate = random.uniform(*learning_rate_range)

    return seqs_per_file, batch_size, epochs, learning_rate
if __name__ == '__main__':

    csv_filename = "results.csv"
    
    # Open the CSV file in write mode and create a CSV writer object
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row to the CSV file
        writer.writerow(["Seed","Seqs per file", "Batch size", "Epochs", "Learning Rate", "Pearson Correlation", "Elapsed Time"])

        # Loop for 20 iterations and write results for each iteration
        for i in range(20):
            start_time = time.time()
            seqs_per_file, batch_size, epochs, learning_rate = generateHyperparams()
            print('Testing Hyperparams:\n', 'Seqs per file:', seqs_per_file, ', Batch size:', batch_size, ', Epochs:', epochs, ', Learning Rate:', learning_rate)
            model = createModel()
            trainModel(model, seqs_per_file, batch_size, epochs, learning_rate)
            pearson_correlation = evaluateModel(model)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed Time: {elapsed_time} seconds")

            # Write the results to the CSV file
            writer.writerow([torch.get_rng_state(),seqs_per_file, batch_size, epochs, learning_rate, pearson_correlation, elapsed_time])

        