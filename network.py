import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Network model predicts disparity on stereo event images
'''

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        
        #dropout = 0.2

        # # left and right channels for disparity estimation - image reduced by 8
        # self.reduction = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
        #     nn.MaxPool2d(kernel_size=8),
        #     nn.BatchNorm1d(num_features=32),
        #     nn.ReLU()
        # )

        # # correspondence network - parallel cnn at increasing dilation rate
        # self.correspondence = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, dilation=1),
        #     nn.Dropout(p=dropout),
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, dilation=2),
        #     nn.Dropout(p=dropout),
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, dilation=3),
        #     nn.Dropout(p=dropout),
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, dilation=4),
        #     nn.Dropout(p=dropout)
        # )

        # # disparity network - dense interconnection inspired by DenseNet
        # self.disparity = nn.Sequential(
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, dilation_rate=1),
        #     nn.Dropout(p=dropout),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, dilation_rate=2),
        #     nn.Dropout(p=dropout),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, dilation_rate=3),
        #     nn.Dropout(p=dropout),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, dilation_rate=4),
        #     nn.Dropout(p=dropout),
        # )

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1,2)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2)
            # nn.Flatten(1),
            # nn.Linear(in_features=593424, out_features=1000), # from 5x5 image dimension
            # nn.ReLU(),
            # nn.Linear(in_features=1000, out_features=1000),
            # nn.ReLU(),
            # nn.Linear(in_features=1000, out_features=307200)
        )

    def forward(self, x):
        return self.layer(x)
    
network = Network()
print(network)
