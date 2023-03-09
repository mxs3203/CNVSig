import torch
from torch import nn


class CNVExtractorAE(nn.Module):

    def __init__(self, input_channels=9, encoded_space_dim=10):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, 12, (4, 4), stride=(1, 1)),
            nn.ReLU6(), nn.Dropout2d(0.2),
            nn.Conv2d(12, 15, (4, 4)),
            nn.ReLU6(),
            nn.Conv2d(15, 20, (4, 4)),
            nn.ReLU6(),
            nn.Conv2d(20, 25, (5, 5)),
            nn.ReLU6(),
            nn.Conv2d(25, 30, (5, 5)),
            nn.ReLU6(),
            nn.Conv2d(30, 35, (5, 5)),
            nn.ReLU6()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(35, 64), nn.ReLU6(),
            nn.Linear(64, encoded_space_dim),nn.ReLU6()
        )
        # decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64), nn.ReLU6(),
            nn.Linear(64, 35),nn.ReLU6()
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(35, 1, 1))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(35, 30, (5, 5), padding=(1,1)),
            nn.ReLU6(),
            nn.ConvTranspose2d(30, 25, (5, 5)),
            nn.ReLU6(),
            nn.ConvTranspose2d(25, 20, (5, 5)),
            nn.ReLU6(),
            nn.ConvTranspose2d(20, 15, (5, 5)),
            nn.PReLU(),
            nn.ConvTranspose2d(15, 12, (5, 5)),
            nn.ReLU6(), nn.Dropout2d(0.2),
            nn.ConvTranspose2d(12, input_channels, (4, 4)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        L = self.encoder_lin(x)
        x = self.decoder_lin(L)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x, L
