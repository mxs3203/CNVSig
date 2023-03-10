from torch import nn

class CNVExtractorAE(nn.Module):

    def __init__(self, input_channels=9, encoded_space_dim=10, conv_scale=1):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, (input_channels-1)*conv_scale, (4, 4), stride=(1, 1)),
            nn.ReLU6(),
            nn.Conv2d( (input_channels-1)*conv_scale,  (input_channels-1)*conv_scale+2, (4, 4)),
            nn.ReLU6(),
            nn.Conv2d( (input_channels-1)*conv_scale+2,  (input_channels-1)*conv_scale+4, (4, 4)),
            nn.ReLU6(),
            nn.Conv2d( (input_channels-1)*conv_scale+4,  (input_channels-1)*conv_scale+6, (5, 5)),
            nn.ReLU6(),
            nn.Conv2d( (input_channels-1)*conv_scale+6,  (input_channels-1)*conv_scale+8, (5, 5)),
            nn.ReLU6(),
            nn.Conv2d( (input_channels-1)*conv_scale+8,  (input_channels-1)*conv_scale+10, (5, 5)),
            nn.ReLU6()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear( (input_channels-1)*conv_scale+10, 512), nn.ReLU6(),
            nn.Linear(512, encoded_space_dim),nn.ReLU6()
        )
        # decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512), nn.ReLU6(),
            nn.Linear(512,  (input_channels-1)*conv_scale+10), nn.ReLU6()
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=( (input_channels-1)*conv_scale+10, 1, 1))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d( (input_channels-1)*conv_scale+10,  (input_channels-1)*conv_scale+8, (5, 5), padding=(1,1)),
            nn.ReLU6(),
            nn.ConvTranspose2d( (input_channels-1)*conv_scale+8,  (input_channels-1)*conv_scale+6, (5, 5)),
            nn.ReLU6(),
            nn.ConvTranspose2d( (input_channels-1)*conv_scale+6,  (input_channels-1)*conv_scale+4, (5, 5)),
            nn.ReLU6(),
            nn.ConvTranspose2d( (input_channels-1)*conv_scale+4,  (input_channels-1)*conv_scale+2, (5, 5)),
            nn.ReLU6(),
            nn.ConvTranspose2d( (input_channels-1)*conv_scale+2,  (input_channels-1)*conv_scale, (5, 5)),
            nn.ReLU6(),
            nn.ConvTranspose2d( (input_channels-1)*conv_scale, input_channels, (4, 4)),
            nn.ReLU6()
        )



    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        L = self.encoder_lin(x)
        x = self.decoder_lin(L)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x, L
