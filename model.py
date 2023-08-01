import torch.nn as nn
import torch.nn.init as init

# Recreation attempt of DeepSELEX design.
# The performance aren't cloes to what the papers says it should be, so something in implementation may be wrong.
class DeepSELEXModel(nn.Module):
    def __init__(self, inputShape = (20,4), classes = 6):
        super(DeepSELEXModel, self).__init__()
        kernelsPerConv = 512
        poolKernelStride = 5
        # First Conv1d layer with kernel size 3
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=8, stride=1, padding=4, bias=True)
        init.normal_(self.conv_layer.weight)
        init.normal_(self.conv_layer.bias)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=poolKernelStride, stride=poolKernelStride)
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(kernelsPerConv * (inputShape[0] // 5), 64),
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
        x = self.flatten(x)
        # Reshape x for fully connected layers
        # x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)
        return x

# Simple Model, just input to hidden layers and outputs classes (based on amount of files given).
# Runs on the same system that runs DeepSELEX, so it compute using a sliding window.
# See pearson.csv DeepModel attempts for performance on RBPs, but in general it runs better than DeepSELEX. 
class DeepModel(nn.Module):
    def __init__(self, inputShape = (20,4), classes = 6):
        super(DeepModel, self).__init__()
        self.hidden_size = 32  # Number of neurons in each hidden layer
        # Define the layers
        self.input_layer = nn.Linear(inputShape[0] * inputShape[1], self.hidden_size)
        self.hidden_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
         # Flatten the input using nn.Flatten()
        x = self.flatten(x)
        
        # Pass through the layers with ReLU activation function using nn.ReLU()
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.output_layer(x)

        # x = self.softmax(x)

        return x

# A trial to use Conv in the model with hidden layers.
# After testing, it seems to get better results in some, reached ~0.4 in some, but some RBPs just doesn't work with it well.
# I think that the longer training set seqs shows lower pearson corr, so it worth looking at that too.
class DeepConvModel(nn.Module):
    def __init__(self, inputShape = (20,4), classes = 6):
        super(DeepConvModel, self).__init__()

        self.hidden_size = 32
        self.conv_chs = 128
        self.kernel_size = 4

        # First Conv1D layer
        self.conv_layer1 = nn.Conv1d(in_channels=inputShape[1], out_channels=self.conv_chs, kernel_size=self.kernel_size, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second Conv1D layer
        self.conv_layer2 = nn.Conv1d(in_channels=self.conv_chs, out_channels=self.conv_chs, kernel_size=self.kernel_size, stride=1, padding=2, bias=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Define the layers
        input_size = self.conv_chs * (inputShape[0] // 4) * inputShape[1] // 4
        self.input_layer = nn.Linear(input_size, self.hidden_size)
        self.hidden_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, classes)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)
        # Conv1D layer 1
        x = self.conv_layer1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        # Conv1D layer 2
        x = self.conv_layer2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))

        # Output layer
        x = self.output_layer(x)

        # x = self.softmax(x)

        return x
