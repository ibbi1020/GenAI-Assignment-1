"""Denoising autoencoder for Tasks 1 and 2.

Fully convolutional. The encoder halves the spatial size at every block and
doubles the channels. The decoder mirrors it. There are no skip connections
and no fully connected layers.
"""

from torch import nn
import math

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for image reconstruction.

    Args:
        encoder_channels (int): Number of channels in the first encoder layer.
        bottleneck_dimension (int): Spatial dimension of the bottleneck layer.
        norm (str): Normalization type - 'group', 'batch', or 'none'.
    """
    def __init__(self, encoder_channels, bottleneck_dimension, norm, *args, **kwargs):
        super().__init__(*args, **kwargs)

        in_channels = 3
        image_dim = 128 
        out_channels = encoder_channels
        encoder_layers = self.get_conv_block(in_channels, out_channels, norm)
        num_stages = int(math.log2(image_dim) - math.log2(bottleneck_dimension)) - 1
        for _ in range(num_stages):
            encoder_layers.extend(self.get_conv_block(out_channels, out_channels*2, norm))
            out_channels *= 2

        decoder_layers = []
        last = False
        for layer in reversed(encoder_layers):
            if layer._get_name() != 'Conv2d':
                continue
            if layer.in_channels == 3:
                last = True
            decoder_layers.extend(self.get_dconv_block(layer.out_channels, layer.in_channels, norm=norm, last=last))
        
        self.dae_e2e = nn.ModuleList(encoder_layers + decoder_layers)
        
    def forward(self, image):
        for layer in self.dae_e2e:
            image = layer(image)

        return image
    
    def get_conv_block(self, in_channels, out_channels, norm):
        assert norm in ['group', 'batch', 'none']
        block = [nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)]
        if norm == 'group':
            block.append(nn.GroupNorm(num_groups=min(out_channels, 8), num_channels=out_channels))
        elif norm == 'batch':
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))
        return block
    
    def get_dconv_block(self, in_channels, out_channels, norm, last=False):
        assert norm in ['group', 'batch', 'none']
        block = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1)]
        if last:
            block.append(nn.Sigmoid())
            return block
        if norm == 'group':
            block.append(nn.GroupNorm(num_groups=8, num_channels=out_channels))
        elif norm == 'batch':
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))
        return block
