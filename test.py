import argparse
from model import Model
# Custom action to enforce RBNS file count constraint
class RBNSCountAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not 4 <= len(values) <= 6:
            raise argparse.ArgumentError(self, "RBNS file count must be between 4 and 6")
        setattr(namespace, self.dest, values)
# Custom argument type to validate RBNS file paths
def rbns_file_type(path):
    # Add your own validation logic if needed
    return path
# NOTE: should be the args needed based on the project submission file.
def recieveArgs():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('outputFile', help='Path to the output file')
    parser.add_argument('RNAcompete', help='Path to the RNAcompete file')
    parser.add_argument('RBNSinput', help='Path to the RBNS_input file')
    parser.add_argument('RBNS', nargs='+', type=rbns_file_type, action=RBNSCountAction,
                        help='Paths to RBNS files (4 to 6 files)')

    # Parse the arguments
    args = parser.parse_args()
    # Access the values of the arguments
    outputFile = args.outputFile
    RNAcompetePath = args.RNAcompete
    RBNSinputPath = args.RBNSinput
    RBNSPaths = args.RBNS
    # Split RBNSPaths into separate variables
    RBNS5nMPath, RBNS20nMPath, RBNS80nMPath, RBNS320nMPath, *additional_RBNSPaths = RBNSPaths
    # Assign additional RBNS file paths to separate variables
    if len(additional_RBNSPaths) == 2:
        RBNS320nMPath, RBNS1300nMPath = additional_RBNSPaths
    elif len(additional_RBNSPaths) == 1:
        RBNS320nMPath = additional_RBNSPaths[0]
        RBNS1300nMPath = None
    else:
        RBNS320nMPath = RBNS1300nMPath = None

    # Do something with the arguments
    print('RNAcompete file:', RNAcompetePath)
    print("RBNS Input file:", RBNSinputPath)
    print("RBNS6 file:", RBNS5nMPath)
    print("RBNS2 file:", RBNS20nMPath)
    print("RBNS3 file:", RBNS80nMPath)
    print("RBNS4 file:", RBNS320nMPath)
    print("RBNS5 file:", RBNS1300nMPath)

    return outputFile, RNAcompetePath, RBNSinputPath, RBNS5nMPath, RBNS20nMPath, RBNS80nMPath, RBNS320nMPath, RBNS1300nMPath


if __name__ == '__main__':
    outputFile, RNAcompetePath, RBNSinputPath, RBNS5nMPath, RBNS20nMPath, RBNS80nMPath, RBNS320nMPath, RBNS1300nMPath = recieveArgs()
    # recieved args, atleast 4 RBNS, up to 6 RBNS.
    NNM = Model()
    NNM.loadModel(outputFile, RNAcompetePath, RBNSinputPath, RBNS5nMPath, RBNS20nMPath, RBNS80nMPath, RBNS320nMPath, RBNS1300nMPath)
    NNM.process()