from torch import nn

class CNVExtractorContrastive(nn.Module):

    def __init__(self, input_channels=9, encoded_space_dim=10, conv_scale=1):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, 16, (2, 2), stride=(2,2), padding=0), nn.LeakyReLU(), # 11
            nn.Conv2d(16, 32, (2, 2), stride=(2,2), padding=0), nn.LeakyReLU(), # 5
            nn.Conv2d(32, 64, (2, 2), stride=(2,2), padding=0), nn.LeakyReLU(), # 2
            nn.Conv2d(64, 128, (2, 2), stride=(2,2), padding=0), nn.LeakyReLU(), # 1
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, encoded_space_dim)
        )
        # decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 1, 1))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), stride=(3, 3)), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(3, 3), output_padding=(1, 1)), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2), output_padding=(1, 1)), nn.LeakyReLU(),
            nn.ConvTranspose2d(16,  input_channels, (2, 2)) #
        )



    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        L = self.encoder_lin(x)
        x = self.decoder_lin(L)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x, L
