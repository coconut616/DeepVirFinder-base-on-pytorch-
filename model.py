import torch
import torch.nn as nn


class DeepVirFinderModel(nn.Module):
    def __init__(self, num_channels=4, num_filters=16, kernel_size=3, pool_size=2,
                 seq_length=150):
        super(DeepVirFinderModel, self).__init__()
        self.conv = nn.Conv1d(num_channels, num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()

        conv_size = (seq_length + 2 * 1 - kernel_size) + 1
        pool_out_size = conv_size // pool_size

        self.fc1 = nn.Linear(num_filters * pool_out_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f"Original input shape: {x.shape}")
        x = self.conv(x)
        print(f"After conv shape: {x.shape}")
        x = self.relu(x)
        x = self.pool(x)
        print(f"After pool shape: {x.shape}")
        x = torch.flatten(x, 1)
        print(f"After flatten shape: {x.shape}")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward_backward_avg(self, fwd, bwd):
        fwd_out = self.forward(fwd)
        bwd_out = self.forward(bwd)
        return (fwd_out + bwd_out) / 2
