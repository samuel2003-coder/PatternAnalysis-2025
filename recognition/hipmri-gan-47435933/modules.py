import torch
import torch.nn as nn

# ----------------------
# Generator (slightly deeper, keeping 64x64 output)
# ----------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, feature_g=64):
        super(Generator, self).__init__()
        # Defining generator as a stack of convolutional transpose layers (upsampling)
        self.model = nn.Sequential(
            # Input: latent vector (latent_dim x 1 x 1)
            # Output: 512 x 4 x 4
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=feature_g*16,  # Slightly more feature maps
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_g*16),
            nn.ReLU(inplace=True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(
                in_channels=feature_g*16,
                out_channels=feature_g*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_g*8),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                in_channels=feature_g*8,
                out_channels=feature_g*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_g*4),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                in_channels=feature_g*4,
                out_channels=feature_g*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_g*2),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                in_channels=feature_g*2,
                out_channels=feature_g,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_g),
            nn.ReLU(inplace=True),

            # Still keeps output 64x64 due to stride=1, padding=1
            nn.ConvTranspose2d(
                in_channels=feature_g,
                out_channels=feature_g,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_g),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ConvTranspose2d(
                in_channels=feature_g,
                out_channels=img_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            # The last layer uses Tanh to scale the output pixel values to the range [-1, 1],
            # matching the normalization of the real images in the dataset.
            nn.Tanh()  # Output in range [-1, 1]
        )

    # Run the input noise vector through the set of convolutional layers defined above and return it as an image of the correct size
    def forward(self, z):
        return self.model(z)


# ----------------------
# Discriminator / Critic (slightly deeper)
# ----------------------
# Downsamples because it extracts features, reducing the image to a single number
class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_d=64):
        super(Discriminator, self).__init__()
        # Convolution block for downsampling
        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=feature_d,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # LeakyReLU instead negative inputs from being zeroed in the discriminator/critic,
            # ensuring gradients flow even for negative activations and improving GAN stability
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(
                in_channels=feature_d,
                out_channels=feature_d*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_d*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 16x16 -> 8x8
            # Only increase the output channels once the image gets small enough at 16x16
            nn.Conv2d(
                in_channels=feature_d*2,
                out_channels=feature_d*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_d*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(
                in_channels=feature_d*4,
                out_channels=feature_d*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Extra depth layer (adds more feature extraction)
            nn.Conv2d(
                in_channels=feature_d*8,
                out_channels=feature_d*8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Last layer is not normalised as we have attained our critic score
            # 4x4 -> 1x1
            nn.Conv2d(
                in_channels=feature_d*8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )

    # view(-1) is a way of flattening the 1D tensor to the according batch size, where each element is the discriminator's score for each image.
    def forward(self, x):
        return self.model(x).view(-1)

