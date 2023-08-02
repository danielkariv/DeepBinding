
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

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
    def __init__(self, file_paths, SEED, nrows_per_file = -1):
        self.file_paths = file_paths
        self.SEED = SEED
        self.nrows_per_file = nrows_per_file

        self.file_seqs, self.file_targets, self.cumulative_lengths = self._load_data()
        print('Seqs loaded: ', len(self.file_seqs))
    def __len__(self):
        return len(self.file_seqs)
    
    def _load_data(self):
        # Load and concatenate data from multiple files
        nucleotides = [index for index in range(len(self.file_paths))]
        data_list = []
        target_list = []
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
            is_duplicate_seqs = True
            if is_duplicate_seqs: # Loads top seqeunces, duplicate them based on the count. (NOTE: May have worse performance, still figure it out.)
                total_added = 0
                # Iterate through each row and add RNA sequences based on their counts
                for _, row in file_data.iterrows():
                    sequence = row['RNA']
                    count = row['Counts']
                    # Add the sequence 'count' number of times to the data_list
                    data_list.extend([sequence] * count)
                    total_added += count
                    # Create truth/target/outputs for the samples, based on the file index.
                    target = [index]
                    target_onehot = one_hot_encoding(target, nucleotides, 1, 0)[0]
                    # Add the one-hot encoded target 'count' number of times to the target_list
                    target_list.extend([target_onehot] * count)
                    if total_added >= self.nrows_per_file:
                        break
                
                # Save lengths for use in _get_file_index (TODO: maybe I don't it, because of targets, or we could use it to make the memory usage lower)
                cumulative_lengths.append(cumulative_lengths[-1] + total_added)
            else: # load the top sequences, ignoring the count. (NOTE: Perforamance still checked)
                data_list.extend(file_data['RNA'])
                count = len(file_data['RNA'])
                target = [index]
                target_onehot = one_hot_encoding(target, nucleotides, 1, 0)[0]
                # Add the one-hot encoded target 'count' number of times to the target_list
                target_list.extend([target_onehot] * count)
                # Save lengths for use in _get_file_index (TODO: maybe I don't it, because of targets, or we could use it to make the memory usage lower)
                cumulative_lengths.append(cumulative_lengths[-1] + count)

            print('Loaded seqs file: ', file_path)
        
        return data_list, target_list, cumulative_lengths

    # TODO: maybe remove it.
    def _get_file_index(self,index):
        file_index = sum(1 for length in self.cumulative_lengths if length <= index)
        return file_index
    
    def getClassesNum(self):
        input_data, target= self.__getitem__(0)
        return len(target)
    
    def getInputShape(self):
        input_data, _ = self.__getitem__(0)
        return input_data.shape

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.file_seqs[index]
        binding_score = self.file_targets[index]
        
        # preprocess the RNA Sequence by creating an one hot encoding, with padding and max length (RBNS has sequences with the same size, but in RNCMPT they are different in size and so is needed)
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence,['A','C','G','T', 'N'],max_length=20,padding_value=0.25)

        # Convert the input/ouput to tensors for usage in pytorch.
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)
        target = torch.tensor(binding_score, dtype=torch.float32)

        return input_data, target

# Custom Dataset to read and process data in chunks
class RNCMPTDataset(Dataset):
    def __init__(self, RNAcompete_sequences_path, RNCMPT_training_path, expectedSeqLength):
        with open(RNAcompete_sequences_path, 'r') as f:
            self.seqs_data = f.readlines()
        
        with open(RNCMPT_training_path, 'r') as f:
            self.targets_data = [float(line.strip()) for line in f.readlines()]

        # NOTE: because training set has different sizes depending on the RBP, we recieve the expected Sequence length, and use it to pad the sequences that we want to run on.
        self.expectedSeqLength = expectedSeqLength
    def __len__(self):
        return len(self.seqs_data)

    def __getitem__(self, index):
        # Access the RNA sequence at the specified index
        rna_sequence = self.seqs_data[index].strip()
        binding_score = torch.tensor(self.targets_data[index], dtype=torch.float32)
        
        # Preprocess the RNA Sequence by creating an one hot encoding, with padding and max length
        preprocessed_rna_sequence = one_hot_encoding(rna_sequence, ['A', 'C', 'G', 'U', 'N'], max_length=self.expectedSeqLength, padding_value=0.25)
        
        # Convert the input/ouput to tensors for usage in pytorch.
        input_data = torch.tensor(preprocessed_rna_sequence, dtype=torch.float32)
        target = torch.tensor(np.array([binding_score]), dtype=torch.float32)

        return input_data, target
