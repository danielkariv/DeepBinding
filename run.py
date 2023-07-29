from logic import processOnce

if __name__ == '__main__':
    # Run over the RBPs data and try the model over the different RBPs (while saving it's results in outputs)
    # TODO: it crashes because it isn't the exact files for each RBPs, should do something like scaning the folder for RBPX_{}.seq files, and send those sorted to the function.
    #       Look at the RBP11.
    num_rbns_files = 16
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

        processOnce(rbns_file_paths_rbp, RNAcompete_sequences_path_rbp, RNCMPT_training_path_rbp, rbp_num)