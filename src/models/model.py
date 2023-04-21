import torch
import torch.nn as nn

from collections import OrderedDict

from typing import Optional

def get_activation(activation: str):
    if activation == 'ReLU':
        return nn.ReLU()
    else:
        raise NotImplementedError

def get_model(model_name, lr=1e-3, device=torch.device('cuda' if torch.cuda.is_available else 'cpu')):
    # Initialize model
    if model_name == 'ConvAutoencoder':
        model = ConvAutoencoder().to(device)
    elif model_name == 'PredictiveConvAutoencoder':
        model = PredictiveConvAutoencoder().to(device)
    else:
        raise NotImplementedError("This model architecture has not been implemented!!!")

    # Create an optimizer object
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Define loss criterion
    recon_criterion         = nn.MSELoss()
    predictive_criterion    = nn.CrossEntropyLoss()

    return model, optimizer, recon_criterion, predictive_criterion

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
        self.pool1  = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.pool1(out)
        return out

class TransposeConvDecoderBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int = 0, activation: Optional = 'ReLU'):
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
        self.act1   = get_activation(activation) if activation is not None else None
    
    def forward(self, z):
        out = self.conv1(z)
        if self.act1 is not None:
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
            ('block4', TransposeConvDecoderBlock(256, 1, kernel_size=2, stride=2, activation=None)),
        ]))

    def forward(self, x):
        # Encode input
        x_ = self.encoder(x)
        z = self.encoder_fc(x_.view(x.shape[0], -1))

        # Decode encoded input
        z_ = self.decoder_fc(z).view(x_.shape)
        x_recon = self.decoder(z_)
    
        return {'x_recon': x_recon, 'z': z}
    
class PredictiveConvAutoencoder(nn.Module):
    def __init__(self):
        super(PredictiveConvAutoencoder, self).__init__()
        
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
            ('block4', TransposeConvDecoderBlock(256, 1, kernel_size=2, stride=2, activation=None)),
        ]))
        
        self.latent_classifier = nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        # Encode input
        x_ = self.encoder(x)
        z = self.encoder_fc(x_.view(x.shape[0], -1))

        # Decode encoded input
        z_ = self.decoder_fc(z).view(x_.shape)
        x_recon = self.decoder(z_)
    
        # Predict from latent representation
        t_logits = self.latent_classifier(z)
    
        return {'x_recon': x_recon, 'z': z, 't_logits': t_logits}
    


class PredictiveEncoder(nn.Module):
    def __init__(self):
        super(PredictiveEncoder, self).__init__()
        
        # setup parts
        self.encoder = nn.Sequential(OrderedDict([
            ('block1', ConvEncoderBlock(1, 256, kernel_size=3, stride=1, padding=1)),
            ('block2', ConvEncoderBlock(256, 128, kernel_size=3, stride=1, padding=1)),
            ('block3', ConvEncoderBlock(128, 64, kernel_size=3, stride=1, padding=1)),
            ('block4', ConvEncoderBlock(64, 32, kernel_size=3, stride=1, padding=1)),
        ]))
        self.encoder_fc = nn.Linear(in_features=32*6*2, out_features=32)        
        self.latent_classifier = nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        # Encode input
        x_ = self.encoder(x)
        z = self.encoder_fc(x_.view(x.shape[0], -1))

        # Predict from latent representation
        t_logits = self.latent_classifier(z)
    
        return {'z': z, 't_logits': t_logits}
    