import torch
import torch.nn as nn

from collections import OrderedDict

def get_activation(activation: str):
    if activation == 'ReLU':
        return nn.ReLU()
    else:
        raise NotImplementedError

def get_model(device, lr = 1e-3):
    # Initialize model
    AE = ConvAutoencoder().to(device)
    # Create an optimizer object
    optimizer = torch.optim.Adam(AE.parameters(), lr=lr)
    # Define loss criterion
    criterion = nn.MSELoss()

    return AE, optimizer, criterion


class ConvEncoderBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int = 0, activation='ReLU'):
        super(ConvEncoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size,
            stride,
            padding,
        )
        self.bn1    = nn.BatchNorm2d(channels_out)
        self.act1   = get_activation(activation)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.pool1(out)
        return out

class TransposeConvDecoderBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int = 0, activation='ReLU'):
        super(TransposeConvDecoderBlock, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(
            channels_in,
            channels_out,
            kernel_size,
            stride,
            padding,
        )

        self.conv1 = nn.ConvTranspose2d(
            channels_in, 
            channels_out, 
            kernel_size=2, 
            stride=2
        )
        self.bn1    = nn.BatchNorm2d(channels_out)
        self.act1   = get_activation(activation) 
    
    def forward(self, z):
        out = self.conv1(z)
        out = self.bn1(out)
        out = self.act1(out)
        return out
    
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # setup parts
        self.encoder = nn.Sequential(OrderedDict([
            ('block1', ConvEncoderBlock(1, 256, kernel_size=3, stride=1, padding=1)),
            ('block2', ConvEncoderBlock(256, 128, kernel_size=3, stride=1, padding=1)),
            ('block3', ConvEncoderBlock(128, 64, kernel_size=3, stride=1, padding=1)),
            ('block4', ConvEncoderBlock(64, 32, kernel_size=3, stride=1, padding=1)),
        ]))
        self.encoder_fc = nn.Linear(in_features=32*6*2, out_features=32)

        self.decoder_fc = nn.Linear(in_features=32, out_features=32*6*2)
        self.decoder = nn.Sequential(OrderedDict([
            ('block1', TransposeConvDecoderBlock(32, 64, kernel_size=2, stride=2)),
            ('block2', TransposeConvDecoderBlock(64, 128, kernel_size=2, stride=2)),
            ('block3', TransposeConvDecoderBlock(128, 256, kernel_size=2, stride=2)),
            ('block4', TransposeConvDecoderBlock(256, 1, kernel_size=2, stride=2)),
        ]))

    def forward(self, x):
        # Encode input
        x_ = self.encoder(x)
        z = self.encoder_fc(x_.view(x.shape[0], -1))

        # Decode encoded input
        z_ = self.decoder_fc(z).view(x_.shape)
        x_recon = self.decoder(z_)
    
        return {'x_recon': x_recon, 'z': z}