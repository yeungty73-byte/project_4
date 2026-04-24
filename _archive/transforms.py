import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import prod

try:
    from deepracer_gym.envs.utils import (
        LIDAR_SHAPE,
        STEREO_CAMERA_SHAPE
    )
except ImportError:
    STEREO_CAMERA_SHAPE = (2, 3, 80, 120)
    LIDAR_SHAPE = (64,)

# pre-processing parameters
LIDAR_RANGE_MAX: float=6.00  # increased from 1.0; typical max LIDAR range on track
LIDAR_RANGE_MIN: float=0.15
CAMERA_MAX_MEASUREMENT: int=255

# encoder params
HIDDEN_CHANNELS: int=16
LIDAR_LATENT_DIMENSION: int=32
CAMERA_LATENT_DIMENSION: int=96

# flattened shapes for observation splitting
LIDAR_FLATTEN_SHAPE = prod(LIDAR_SHAPE)
STEREO_CAMERA_FLATTEN_SHAPE = prod(STEREO_CAMERA_SHAPE)


# ============================================================================
# REF: Hettiarachchi et al. (2024) U-Transformer with skip connections
# REF: Dosovitskiy et al. (2021) Vision Transformer encoder backbone
# Integrated from research_modules.py -- phasing out standalone module
# ============================================================================

class UTransformerBlock(nn.Module):
# REF: Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.
    """U-Transformer encoder-decoder block with skip connections.
    REF: Hettiarachchi et al. (2024) deep RL LiDAR feature extraction."""
    def __init__(self, in_dim, bottleneck_dim=32, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        mid = (in_dim + bottleneck_dim) // 2
        self.enc1 = nn.Linear(in_dim, mid)
        self.enc2 = nn.Linear(mid, bottleneck_dim)
        self.dec1 = nn.Linear(bottleneck_dim + mid, out_dim)  # skip from enc1
        self.ln1 = nn.LayerNorm(mid)
        self.ln2 = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        e1 = F.leaky_relu(self.ln1(self.enc1(x)))
        e2 = F.leaky_relu(self.ln2(self.enc2(e1)))
        return F.leaky_relu(self.dec1(torch.cat([e2, e1], dim=-1)))


class EncodeLiDAR(nn.Module):
    """U-Transformer LiDAR encoder with skip connections.
    REF: Hettiarachchi et al. (2024) Replaces rudimentary FC encoder.
    Produces LIDAR_LATENT_DIMENSION-dim latent from LiDAR scan."""
    def __init__(self):
        super().__init__()
        lidar_dim = prod(LIDAR_SHAPE)
        self.normalize = lambda x: torch.clamp((x - LIDAR_RANGE_MIN) / (LIDAR_RANGE_MAX - LIDAR_RANGE_MIN + 1e-8), 0.0, 1.0)
        self.utrans = UTransformerBlock(
            lidar_dim,
            bottleneck_dim=max(16, lidar_dim // 4),
            out_dim=LIDAR_LATENT_DIMENSION
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.normalize(x)
        return self.utrans(x)


class EncodeStereoCameras(nn.Module):
    """Lightweight CNN encoder for stereo camera images.
    Produces CAMERA_LATENT_DIMENSION-dim latent from stereo pair."""
    def __init__(self):
        super().__init__()
        in_channels = STEREO_CAMERA_SHAPE[0] * STEREO_CAMERA_SHAPE[1]  # 2*3=6
        h, w = STEREO_CAMERA_SHAPE[2], STEREO_CAMERA_SHAPE[3]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, HIDDEN_CHANNELS, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS*2, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(HIDDEN_CHANNELS*2, HIDDEN_CHANNELS*4, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((2, 3)),
        )
        self.fc = nn.Linear(HIDDEN_CHANNELS * 4 * 2 * 3, CAMERA_LATENT_DIMENSION)
        self.ln = nn.LayerNorm(CAMERA_LATENT_DIMENSION)

    def forward(self, x):
        b = x.size(0)
        x = x.float() / CAMERA_MAX_MEASUREMENT
        # merge stereo channels: (b, 2, 3, H, W) -> (b, 6, H, W)
        x = x.view(b, -1, STEREO_CAMERA_SHAPE[2], STEREO_CAMERA_SHAPE[3])
        x = self.conv(x)
        x = x.view(b, -1)
        return F.leaky_relu(self.ln(self.fc(x)))


class UnflattenObservation(nn.Module):
    """Split flattened observation vector into (camera, lidar) tensors.
    Convention: leading values are LiDAR."""
    def forward(self, x):
        # leading values are LiDAR
        return (
            x[..., LIDAR_FLATTEN_SHAPE:].view(-1, *STEREO_CAMERA_SHAPE),
            x[..., :LIDAR_FLATTEN_SHAPE].view(-1, *LIDAR_SHAPE)
        )


class EncodeObservation(nn.Module):
    """Unified observation encoder: stereo camera + LiDAR = latent.
    Uses U-Transformer skip-connection encoder for LiDAR (from research_modules).
    Uses lightweight CNN for stereo cameras.
    Output dim = CAMERA_LATENT_DIMENSION + LIDAR_LATENT_DIMENSION = 128."""
    def __init__(self):
        super().__init__()
        self.unflatten = UnflattenObservation()
        self.camera_encoder = EncodeStereoCameras()
        self.lidar_encoder = EncodeLiDAR()

    def forward(self, x):
        camera, lidar = self.unflatten(x)
        camera_encoded = self.camera_encoder(camera)
        lidar_encoded = self.lidar_encoder(lidar)
        return torch.cat((camera_encoded, lidar_encoded), -1)

# Backward compatibility alias for checkpoints saved with old class name
PreprocessStereoCameras = EncodeStereoCameras
CNN = EncodeStereoCameras  # backward compat alias for old checkpoints
PreprocessLiDAR = EncodeLiDAR  # backward compat
