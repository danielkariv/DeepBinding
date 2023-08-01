import torch
import torch.nn as nn
# This script contains the code of all the models I tried.
# Going from the top-first to bottom-last.
# There are notes on how each models compare with past models.
# The outputs are saved in folders in the outputs_(MODELNAME).


# NOTE: based on DeepSELEX (without the output cycles and sliding window).
#       Time: 200 seconds per RBPs, Performance: correlation of 0.00-0.15.
#       Issue: most RBPs has correlation close to 0. 
#       (seqs_per_file, batch_size, epochs, learning_rate = 50000, 32, 3, 0.001)
class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()
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

# NOTE: based on ModelV1, but adds:
#       - padding to the input at the conv layer.
#       - Dropouts.
#       - ReLU at the layer before the last one.
#       Time: 220 seconds per RBPs, Performance: most has correlation of 0.04-0.18. 
#       Issue: RBP7 has -0.027 correlation. RBP8 has ~0 correlation.
#       (seqs_per_file, batch_size, epochs, learning_rate = 50000, 32, 3, 0.001)
class ModelV2(nn.Module):
    def __init__(self):
        super(ModelV2, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=8, stride=1, padding=4)  # Correct padding
        self.maxpool_layer = nn.MaxPool1d(kernel_size=5, stride=5)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8, 64), # NOTE: input is based on size of maxpool (512,8).
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.ReLU(),
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
# NOTE: based on ModelV2, but changes:
#      - change kernel and padding to kernel=5, padding=3.
#      It's a tiny bit better, but not much improvement.
#       (seqs_per_file, batch_size, epochs, learning_rate = 50000, 32, 3, 0.001)
class ModelV3(nn.Module):
    def __init__(self):
        super(ModelV3, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=5, stride=1, padding=3)  # Correct padding
        self.maxpool_layer = nn.MaxPool1d(kernel_size=5, stride=5)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8, 64), # NOTE: input is based on size of maxpool (512,8).
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.ReLU(),
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
# NOTE: It similar to past models but now with:
#      Multiple Conv1D layers, runs in parallels, and combined into a fully connected layers.
#      The idea is to capture more k-mers: 3-mers, 5-mers, and 8-mers.
#      Time: 230 seconds. 
#      Performance: Correlation have been improve on almost all RBPs (expect RBP7, which was not good either before).
#      (seqs_per_file, batch_size, epochs, learning_rate = 50000, 32, 3, 0.001)
class ModelV4(nn.Module):
    def __init__(self):
        super(ModelV4, self).__init__()
        kernelsPerConv = 64
        # First Conv1d layer with kernel size 3
        self.conv_layer_3 = nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=3, stride=1, padding=1)
        self.relu_3 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool1d(kernel_size=5, stride=5)

        # Second Conv1d layer with kernel size 5
        self.conv_layer_5 = nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=5, stride=1, padding=2)
        self.relu_5 = nn.ReLU()
        self.maxpool_5 = nn.MaxPool1d(kernel_size=5, stride=5)

        # Third Conv1d layer with kernel size 8
        self.conv_layer_8 = nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=8, stride=1, padding=3)
        self.relu_8 = nn.ReLU()
        self.maxpool_8 = nn.MaxPool1d(kernel_size=5, stride=5)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(kernelsPerConv * 3 * 8, 128),  # input size = 1536 is the number of output channels from each Conv1d layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)

        # Apply the three Conv1d layers with different kernel sizes
        x_3 = self.conv_layer_3(x)
        x_3 = self.relu_3(x_3)
        x_3 = self.maxpool_3(x_3)

        x_5 = self.conv_layer_5(x)
        x_5 = self.relu_5(x_5)
        x_5 = self.maxpool_5(x_5)

        x_8 = self.conv_layer_8(x)
        x_8 = self.relu_8(x_8)
        x_8 = self.maxpool_8(x_8)

        # Concatenate the outputs from the three Conv1d layers
        x = torch.cat((x_3, x_5, x_8), dim=2)

        # Reshape x for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)
        return x
    
# NOTE: It expended version of ModelV4, with more kernels sizes.
#       I also scale up the hidden layers.
#       And scale up the amount of kernels per Conv1D.
#       Time: 400 seconds.
#       Performance: It works similarly, some small improvments in some RBPs.
#       Issues: Some (around 3-5) RBPs return loss 0.3 which prints in the end 1.0 and return correlations Nan.
#       Because of the Nan, I will drop this model idea for now. Goes back to ModelV4 and try things on it.
class ModelV5(nn.Module):
    def __init__(self):
        super(ModelV5, self).__init__()
        kernelsPerConv = 256

        # Conv1d layers with kernel sizes ranging from 3 to 12
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=k, stride=1, padding=k // 2) for k in range(3, 13)
        ])

        # ReLU activation
        self.relu = nn.ReLU()

        # MaxPool1d layers
        self.pool_layers = nn.ModuleList([
            nn.MaxPool1d(kernel_size=5, stride=5) for _ in range(10)  # We have 10 Conv1d layers
        ])

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(kernelsPerConv * 10 * 8, 1028),  # input size is 5120.
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1028, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)

        # Apply Conv1d layers with different kernel sizes
        conv_outputs = []
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            out = pool_layer(self.relu(conv_layer(x)))
            conv_outputs.append(out)

        # Concatenate the outputs from the Conv1d layers
        x = torch.cat(conv_outputs, dim=1)

        # Flatten the tensor to feed it into the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply the fully connected layers
        x = self.fc_layers(x)
        return x

# NOTE: It is model V4 with some changes:
#       - convert the model to look like V5 with dynamic building the conv/pools by a variable (will allow testing later on)
#       - more Kernels Per Conv.     
#       Time: 220 seconds. 
#      Performance: Similar to V4. I think it worth to try looking for other hyperparams.
#      (seqs_per_file, batch_size, epochs, learning_rate = 50000, 32, 3, 0.001)
class ModelV6(nn.Module):
    def __init__(self):
        super(ModelV6, self).__init__()
        kernelsPerConv = 128
        kernelsSizes =  [3,5,8]
        # Conv1d layers with kernel sizes ranging from 3 to 12
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=k, stride=1, padding=k // 2) for k in kernelsSizes
        ])

        # ReLU activation
        self.relu = nn.ReLU()

        # MaxPool1d layers
        self.pool_layers = nn.ModuleList([
            nn.MaxPool1d(kernel_size=5, stride=5) for _ in range(len(kernelsSizes))
        ])

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(kernelsPerConv * len(kernelsSizes) * 8, 1028),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1028, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)

        # Apply Conv1d layers with different kernel sizes
        conv_outputs = []
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            out = pool_layer(self.relu(conv_layer(x)))
            conv_outputs.append(out)

        # Concatenate the outputs from the Conv1d layers
        x = torch.cat(conv_outputs, dim=1)

        # Reshape x for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)
        return x

# NOTE: It is model V6 with some changes:
#       - Kernel sizes changed to 5,7,9,11 (based on reserachs and codebases it seems like a good k-mers to test on)
#       - less Kernels Per Conv (like ModelV4)
#       - Changed the hidden layers to be based on the Kernels per Conv number. 
#       Time: 220 seconds. 
#      Performance: Similar to V4. I think it worth to try looking for other hyperparams.
#      (seqs_per_file, batch_size, epochs, learning_rate = 16000, 128, 5, 0.001)
class ModelV7(nn.Module):
    def __init__(self):
        super(ModelV7, self).__init__()
        kernelsPerConv = 64
        kernelsSizes =  [5, 7, 9, 11]
        # Conv1d layers with kernel sizes ranging from 3 to 12
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=4, out_channels=kernelsPerConv, kernel_size=k, stride=1, padding=k // 2) for k in kernelsSizes
        ])

        # ReLU activation
        self.relu = nn.ReLU()

        # MaxPool1d layers
        self.pool_layers = nn.ModuleList([
            nn.MaxPool1d(kernel_size=5, stride=5) for _ in range(len(kernelsSizes))
        ])

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(kernelsPerConv * len(kernelsSizes) * 8, kernelsPerConv*4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(kernelsPerConv*4, kernelsPerConv),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(kernelsPerConv, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # transpose the matrix
        x = x.permute(0, 2, 1)

        # Apply Conv1d layers with different kernel sizes
        conv_outputs = []
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            out = pool_layer(self.relu(conv_layer(x)))
            conv_outputs.append(out)

        # Concatenate the outputs from the Conv1d layers
        x = torch.cat(conv_outputs, dim=1)

        # Reshape x for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)
        return x
