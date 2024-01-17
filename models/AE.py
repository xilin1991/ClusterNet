import einops

import torch
import torch.nn as nn

from models.modules import activation_fn


class AE(nn.Module):
    def __init__(self, arch, n_input, n_hidden):
        super(AE, self).__init__()
        self.norm_type = 'inr'
        self.encoder_dims = arch[-1]

        self.encoder_cnn, ratio = self.make_encoder(n_input, arch)

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.encoder_dims, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.spatial_attn = nn.Conv2d(n_hidden * 2, n_hidden, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder_cnn = self.make_decoder(n_hidden, n_input, decoder_dims=64)

    def make_encoder(self, in_channels, encoder_arch):
        layers = []
        down_factor = 0
        for v in encoder_arch:
            if v == 'MP':
                layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                down_factor += 1
            else:
                conv1 = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv2 = nn.Conv2d(v, v, kernel_size=3, padding=1)

                layers += [conv1, activation_fn(v, norm_type=self.norm_type, affine=True),
                           conv2, activation_fn(v, norm_type=self.norm_type, affine=True)]
                in_channels = v
        return nn.Sequential(*layers), 2 ** down_factor

    def make_decoder(self, in_channels, out_channels, decoder_dims=64):
        layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, decoder_dims, kernel_size=3, padding=1, output_padding=1, stride=2),
            activation_fn(decoder_dims, norm_type=self.norm_type, affine=True),

            nn.ConvTranspose2d(decoder_dims, decoder_dims, kernel_size=3, padding=1, output_padding=1, stride=2),
            activation_fn(decoder_dims, norm_type=self.norm_type, affine=True),

            nn.Conv2d(decoder_dims, decoder_dims, kernel_size=3, padding=1),
            activation_fn(decoder_dims, norm_type=self.norm_type, affine=True),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        return layers

    def forward(self, x):
        # Encoder
        enc_out = self.encoder_cnn(x)

        # Z Layer
        h, w = enc_out.shape[2:]
        z = einops.rearrange(enc_out, 'b c h w -> b (h w) c')
        z = self.hidden_layer(z)
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)

        # Spatial Attention
        attn_avg = self.avg_pool(z)
        attn_max = self.max_pool(z)
        attn = self.relu(self.spatial_attn(torch.cat([attn_avg, attn_max], dim=1)))
        attn = self.upsample(attn)
        z = z + attn

        # Decoder
        x_bar = self.decoder_cnn(z)

        return x_bar, z


class AE_NS(nn.Module):
    def __init__(self, arch, n_input, n_hidden):
        super(AE_NS, self).__init__()
        self.norm_type = 'inr'
        self.encoder_dims = arch[-1]

        self.encoder_cnn, ratio = self.make_encoder(n_input, arch)

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.encoder_dims, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.spatial_attn = nn.Conv2d(n_hidden * 2, n_hidden, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder_cnn = self.make_decoder(n_hidden, n_input, decoder_dims=64)

    def make_encoder(self, in_channels, encoder_arch):
        layers = []
        down_factor = 0
        for v in encoder_arch:
            if v == 'MP':
                layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                down_factor += 1
            else:
                conv1 = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv2 = nn.Conv2d(v, v, kernel_size=3, padding=1)

                layers += [conv1, activation_fn(v, norm_type=self.norm_type, affine=True),
                           conv2, activation_fn(v, norm_type=self.norm_type, affine=True)]
                in_channels = v
        return nn.Sequential(*layers), 2 ** down_factor

    def make_decoder(self, in_channels, out_channels, decoder_dims=64):
        layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, decoder_dims, kernel_size=3, padding=1, output_padding=1, stride=2),
            activation_fn(decoder_dims, norm_type=self.norm_type, affine=True),

            nn.ConvTranspose2d(decoder_dims, decoder_dims, kernel_size=3, padding=1, output_padding=1, stride=2),
            activation_fn(decoder_dims, norm_type=self.norm_type, affine=True),

            nn.Conv2d(decoder_dims, decoder_dims, kernel_size=3, padding=1),
            activation_fn(decoder_dims, norm_type=self.norm_type, affine=True),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        return layers

    def forward(self, x):
        # Encoder
        enc_out = self.encoder_cnn(x)

        # Z Layer
        h, w = enc_out.shape[2:]
        z = einops.rearrange(enc_out, 'b c h w -> b (h w) c')
        z = self.hidden_layer(z)
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)

        # Spatial Attention
        attn_avg = self.avg_pool(z)
        attn_max = self.max_pool(z)
        attn = self.relu(self.spatial_attn(torch.cat([attn_avg, attn_max], dim=1)))
        attn = self.upsample(attn)
        z = z + attn

        # Decoder
        x_bar = self.decoder_cnn(z)

        return x_bar, z
