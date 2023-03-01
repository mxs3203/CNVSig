import torch
from torch import nn


class CNVExtractor(nn.Module):

    def __init__(self, encoded_space_dim=10):
        super().__init__()
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.BatchNorm2d(22),
            nn.Conv2d(22, 33, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(33, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoded_space_dim)
        )

        # decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(256, 1, 1))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (3, 3) ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 22, (4, 4)),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.encoder_lin(x)
        sigma = torch.exp(self.encoder_lin(x))
        L = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        x = self.decoder_lin(L)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x, L
