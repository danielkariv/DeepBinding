import os
import argparse
from model import Model
# TODO: not sure about what args needed here.
def recieveArgs():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    # Add arguments to the parser
    parser.add_argument('RNAcompete', help='Path to the RNAcompete file')
    parser.add_argument('RBNS_training', help='Path to the RBNS_training dir')
    parser.add_argument('RNCMPT_training', help='Path to the RNCMPT_training dir')
    # Parse the arguments
    args = parser.parse_args()
    # Access the values of the arguments
    RNAcompetePath = args.RNAcompete
    RBNS_trainingPath = args.RBNS_training
    RNCMPT_trainingPath = args.RNCMPT_training
    # Do something with the arguments
    print('RNAcompete file:', RNAcompetePath)
    print("RBNS_training dir:", RBNS_trainingPath)
    print("RNCMPT_training dir:", RNCMPT_trainingPath)
    return RNAcompetePath, RBNS_trainingPath, RNCMPT_trainingPath


if __name__ == '__main__':
    RNAcompetePath, RBNS_trainingPath, RNCMPT_trainingPath = recieveArgs()
    # recieved args, atleast 4 RBNS, up to 6 RBNS.
    NNM = Model()
    with open(RNAcompetePath, 'r') as file:
        contents = file.read()
        print(contents)
    # Iterate over files in the directory
    for filename in os.listdir(RBNS_trainingPath):
        # Construct the file path
        file_path = os.path.join(RBNS_trainingPath, filename)
        # Check if the path is a file
        if os.path.isfile(file_path):
            # Perform operation on the file
            break

    NNM.saveModel()
    
    