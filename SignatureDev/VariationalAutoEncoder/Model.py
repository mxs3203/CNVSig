import torch
from torch import nn


class CNVExtractorVAE(nn.Module):

    def __init__(self, input_channels=9, encoded_space_dim=10):
        super().__init__()
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
            nn.Linear(512, encoded_space_dim)
        )
        self.encoder_lin2 = nn.Sequential(
            nn.Linear(512,encoded_space_dim)
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(encoded_space_dim, 1, 1))
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
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu,log_var = self.encoder_lin(x), self.encoder_lin2(x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
#        x = self.unflatten(z)
#        x = self.decoder_conv(x)
        return x, z,self.log_scale,self.kl_divergence(z, mu, std)
