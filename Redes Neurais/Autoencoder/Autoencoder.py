import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoding layers
        self.encoder_conv1 = nn.Conv2d(3, 32, 2, 1)
        self.encoder_bn1 = nn.BatchNorm2d(32)
        self.encoder_conv2 = nn.Conv2d(32, 16, 2, 1)
        self.encoder_bn2 = nn.BatchNorm2d(16)
        self.encoder_conv3 = nn.Conv2d(16, 3, 2, 2)
        self.encoder_bn3 = nn.BatchNorm2d(3)

        # Decoding layers
        self.decoder_deconv1 = nn.ConvTranspose2d(3, 16, 2, 2)
        self.decoder_bn1 = nn.BatchNorm2d(16)
        self.decoder_deconv2 = nn.ConvTranspose2d(16, 32, 2, 1)
        self.decoder_bn2 = nn.BatchNorm2d(32)
        self.decoder_deconv3 = nn.ConvTranspose2d(32, 3, 2, 1)
        self.decoder_bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = F.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = F.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x = F.relu(self.encoder_bn3(self.encoder_conv3(x)))
        return x
    
    def decode(self, x):
        x = F.relu(self.decoder_bn1(self.decoder_deconv1(x)))
        x = F.relu(self.decoder_bn2(self.decoder_deconv2(x)))
        x = F.relu(self.decoder_bn3(self.decoder_deconv3(x)))
        return x
