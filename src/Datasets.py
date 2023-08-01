
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def preprocess_rna(rna_sequence, nucleotides = ['A', 'C', 'G', 'T']):
    return one_hot_encoding(rna_sequence, nucleotides) # TODO: maybe replace it with auto selecting based on if it is RBNS/RNAcompete seqs. 
def one_hot_encoding(rna_sequence, nucleotides = ['A', 'C', 'G', 'T'], max_length=42, padding_value=0.25):
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
        data_list = []
        target_list = []
        cumulative_lengths = [0]  # Store the cumulative lengths of the files
        for index in range(len(self.file_paths)):
            file_path = self.file_paths[index]
            if self.nrows_per_file < 0:
                self.nrows_per_file = None
            file_data = pd.read_csv(file_path, delimiter='\t', header=None, nrows=self.nrows_per_file)
            
            data_list.append(file_data)
            # target_list.extend( for _ in range(len(file_data))])
            cumulative_lengths.append(cumulative_lengths[-1] + len(file_data))
            print('Loaded seqs file: ', file_path)
        return pd.concat(data_list, ignore_index=True), target_list, cumulative_lengths

    def _get_file_index(self,index):
        file_index = sum(1 for length in self.cumulative_lengths if length <= index)
        return file_index

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.file_seqs.iloc[index, 0]
        binding_score = np.array([self._get_file_index(index)/len(self.file_paths)]) # self.file_targets[index]
        # Preprocess the RNA sequence (if needed)
        preprocessed_rna_sequence = preprocess_rna(rna_sequence)

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
        preprocessed_rna_sequence = preprocess_rna(rna_sequence, ['A', 'C', 'G', 'U'])

        # Convert RNA sequence to tensor
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)

        # Set the target variable.
        target = torch.tensor(np.array([binding_score]), dtype=torch.float32)

        return input_data, target