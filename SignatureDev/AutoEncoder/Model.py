import torch
from torch import nn


class CNVExtractorAE(nn.Module):

    def __init__(self, input_channels=9, encoded_space_dim=10):
        super().__init__()


        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, 16, (4, 4), stride=(1, 1)),
            nn.Sigmoid(), nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, (4, 4)),
            nn.Sigmoid(),
            nn.Conv2d(32, 64, (4, 4)),
            nn.Sigmoid(),
            nn.Conv2d(64, 128, (5, 5)),
            nn.Sigmoid(),
            nn.Conv2d(128, 256, (5, 5)),
            nn.Sigmoid(),
            nn.Conv2d(256, 512, (5, 5)),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid(),
            nn.Linear(128, encoded_space_dim)
        )
        # decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(512, 1, 1))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (5, 5), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.Sigmoid(), nn.Dropout2d(0.2),
            nn.ConvTranspose2d(256, 128, (5, 5)),
            nn.Sigmoid(),
            nn.ConvTranspose2d(128, 64, (5, 5)),
            nn.Sigmoid(),
            nn.ConvTranspose2d(64, 32, (5, 5)),
            nn.Sigmoid(),
            nn.ConvTranspose2d(32, 16, (5, 5)),
            nn.Sigmoid(),
            nn.ConvTranspose2d(16, input_channels, (4, 4)),
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
