import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# Simple Model, just input to hidden layers and outputs classes (based on amount of files given).
# NOTE: Correlation: 0.08631190844959596 (file: pearson_2023-08-21_13-52-10.csv)
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
         # Flatten the input using nn.Flatten()
        x = self.flatten(x)
        
        # Pass through the layers with ReLU activation function using nn.ReLU()
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.output_layer(x)

        x = self.softmax(x)

        return x


# Model is using two Conv1D, one after the another, and then FC layers.
# NOTE: Correlation: 0.07986798011516515 (file: pearson_2023-08-21_15-00-06.csv)
class DeepConvModel(nn.Module):
    def __init__(self, inputShape = (20,4), classes = 6):
        super(DeepConvModel, self).__init__()

        self.hidden_size = 32
        self.conv_chs = 128
        self.kernel_size = 4

        # First Conv1D layer
        self.conv_layer1 = nn.Conv1d(in_channels=inputShape[1], out_channels=self.conv_chs, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second Conv1D layer
        self.conv_layer2 = nn.Conv1d(in_channels=self.conv_chs, out_channels=self.conv_chs, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # Define the layers
        input_size = 1408 # self.conv_chs * (inputShape[0] // 4)
        self.input_layer = nn.Linear(input_size, self.hidden_size)
        self.hidden_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

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
        x = self.flatten(x)

        # Fully connected layers
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))

        # Output layer
        x = self.output_layer(x)
        x = self.softmax(x)

        return x
   
# NOTE: Still doesn't work.. I honestly feel like I miss something here.. Try to redesign it to maybe fit, but it still isn't as good as the other models.
# Recreation attempt of DeepSELEX design.
# The performance aren't cloes to what the papers says it should be, so something in implementation may be wrong.
# NOTE: Correlation: XXX (file: XXX.csv)
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

# Second recreation, still doesn't work as expected..
# NOTE: Correlation: XXX (file: XXX.csv)
class DeepSELEX2(nn.Module):
    def __init__(self, inputShape = (20,4), classes = 6):
        super(DeepSELEX2, self).__init__()
        # TODO: Loading the data right now is taking 15000 samples randomly, but the paper sees to take the most frequent ones.
        #       Need a better loader of information.
        self.hidden_size = 32
        self.conv_chs = 512
        self.kernel_size = 8

        # First Conv1D layer
        self.conv_layer = nn.Conv1d(in_channels=inputShape[1], out_channels=self.conv_chs, kernel_size=self.kernel_size, stride = 1, padding= self.kernel_size//2 , bias=True)
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride = None, padding=0, dilation=1, ceil_mode=False)
        
        # Define the layers
        input_size =  self.conv_chs * 4 * 2 # 4 = RNA letters, * 2(maybe because of input size 41~2*20?).
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_size, 64 )
        self.hidden_layer1 = nn.Linear(64, 32)
        self.hidden_layer2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.65)


    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)
        # Conv1D layer 1
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten the tensor
        x = self.flatten(x)

        # Fully connected layers
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer2(x))
        x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)

        x = self.sigmoid(x)

        return x

# Try Transformer based on ChatGPT.
# Run it a few times only at the first ones. Don't see good scores here.
# NOTE: Correlation: XXX (file: XXX.csv)
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

# Trying to do Multi Conv1D layers at once, and then sending them to hidden layers
# NOTE: Correlation: 0.09537825556168941 (file: pearson_2023-08-21_15-53-15.csv)
class DeepMultiConvModel(nn.Module):
    def __init__(self, inputShape=(20, 4), classes=6):
        super(DeepMultiConvModel, self).__init__()

        self.hidden_size = 32
        self.conv_chs = 128

        # First Conv1D layer
        self.conv_layer1 = nn.Conv1d(in_channels=inputShape[1], out_channels=self.conv_chs, kernel_size=5, stride=1, padding=3, bias=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second Conv1D layer
        self.conv_layer2 = nn.Conv1d(in_channels=inputShape[1], out_channels=self.conv_chs, kernel_size=8, stride=1, padding=4, bias=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Third Conv1D layer
        self.conv_layer3 = nn.Conv1d(in_channels=inputShape[1], out_channels=self.conv_chs, kernel_size=11, stride=1, padding=6, bias=True)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # Define the layers
        input_size = 8064 # self.conv_chs * (inputShape[0] // 8) * 3  # Adjust for three Conv1d layers
        self.hidden_layer = nn.Linear(input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Transpose the matrix
        x = x.permute(0, 2, 1)
        
        # Conv1D layer 1
        x1 = self.conv_layer1(x)
        x1 = self.relu(x1)
        x1 = self.maxpool1(x1)

        # Conv1D layer 2
        x2 = self.conv_layer2(x)
        x2 = self.relu(x2)
        x2 = self.maxpool2(x2)

        # Conv1D layer 3
        x3 = self.conv_layer3(x)
        x3 = self.relu(x3)
        x3 = self.maxpool3(x3)

        # Flatten and concatenate
        x_combined = torch.cat((self.flatten(x1), self.flatten(x2), self.flatten(x3)), dim=1)

        # Fully connected layers
        x_combined = self.relu(self.hidden_layer(x_combined))

        # Output layer
        x_combined = self.output_layer(x_combined)

        return x_combined

# Taking only max value for each kernel, and then running on them with FC layers.
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
# Taking only max value for each kernel, and then running on them with FC layers.
# Also adds Dropout to FC layers.
# NOTE: Correlation: 0.161019015436316/0.15509259165153524 (files: pearson_2023-08-21_04-44-19.csv, pearson_2023-08-21_20-19-46.csv)
class DeepConvModel5(nn.Module):
    def __init__(self, inputShape=(20, 4), classes=6, hidden_size=128, conv_chs=512, kernel_size=4, dropout_prob=0.5):
        super(DeepConvModel5, self).__init__()
        # First Conv1D layer
        self.conv_layer1 = nn.Conv1d(in_channels=inputShape[1], out_channels=conv_chs, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.flatten = nn.Flatten()
        # Define the layers
        input_size = conv_chs
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)
        # Conv1D layer 1
        x = self.conv_layer1(x)
        x, _ = torch.max(x, dim=2)  # along the time dimension
        x = self.relu(x)

        # Flatten the tensor
        x = self.flatten(x)

        # Fully connected layers with dropout
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer2(x))
        x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)
        x = self.softmax(x)

        return x

# Model based on the paper "Deep Learning for Sequence Pattern Recognition"
# Runs two Conv1D, with max pool, and then LSTM, and then two FC.
# NOTE: Correlation: 0.07490990727552513 (file: pearson_2023-08-21_17-09-51.csv)
class CNNRNNModel(nn.Module):
    def __init__(self, input_size=(20, 4), num_classes = 6, hidden_size = 64):
        super(CNNRNNModel, self).__init__()
        convs_channel = 128
        kernel_size = 3
        self.conv1 = nn.Conv1d(in_channels=input_size[1], out_channels=convs_channel//2, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=convs_channel//2, out_channels=convs_channel, kernel_size=kernel_size, padding=kernel_size//2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=convs_channel, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, num_classes)  # Change the output size
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multiclass

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change dimensions for Conv1d
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)  # Change dimensions for LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        x = self.fc1(lstm_out)
        x = self.fc2(x)
        x = self.softmax(x)  # Use softmax for multiclass
        return x

# NOTE: Correlation: 0.14762343832227673 (file: pearson_2023-08-21_22-47-44.csv)
class DeepConvModel6(nn.Module):
    def __init__(self, inputShape=(20, 4), classes=6, hidden_size=256, conv_chs=512, kernel_size=6, dropout_prob=0.25):
        super(DeepConvModel6, self).__init__()
        # First Conv1D layer
        self.conv_layer1 = nn.Conv1d(in_channels=inputShape[1], out_channels=conv_chs, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.flatten = nn.Flatten()
        # Define the layers
        input_size = conv_chs
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)
        # Conv1D layer 1
        x = self.conv_layer1(x)
        x, _ = torch.max(x, dim=2)  # along the time dimension
        x = self.relu(x)

        # Flatten the tensor
        x = self.flatten(x)

        # Fully connected layers with dropout
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer2(x))
        x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)
        x = self.softmax(x)

        return x
