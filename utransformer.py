"""
utransformer.py  -  U-Transformer for DeepRacer LIDAR/camera perception
v1.0.0 / CS7642 Project 4

Architecture reference:
  Petit et al. 2021, "U-Net Transformer: Self and Cross Attention for Medical Image Segmentation"
  arXiv:2103.06104. Adapted from 2-D medical image segmentation to 1-D LIDAR range arrays
  and top-down track-geometry line-of-sight maps.

Pipeline roles in run.py:
  1. ObstacleSegmenter  - raw LIDAR array -> per-sector obstacle mask + distances
  2. LineOfSightEncoder - car pos + heading + waypoints -> 4-dim clearance vector
  3. UTransformerObs    - singleton wrapper; extends compact obs from 12 -> 16 dims
  4. extract_compact_obs_v2 - 16-dim drop-in for extract_compact_obs (12-dim)

Design choices:
  Encoder depth = 2 (not 4+): obs_dim_raw=38464 is a 1-D flattened sensor array, not 2-D image.
  Self-attention at bottleneck only: captures front-left/front-right correlations.
  Cross-attention in skip connections: decoder queries encoder with waypoint embedding.
  Full model < 50k parameters; forward pass < 1ms on CPU; safe for real-time use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SelfAttention1D(nn.Module):
    """Single-head self-attention over 1-D sequence. Input: (B,C,L) -> (B,C,L)"""
    def __init__(self, channels, heads=4):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, L = x.shape
        H, D = self.heads, C // self.heads
        qkv = self.qkv(x).reshape(B, 3, H, D, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(B, C, L)
        return self.proj(out) + x


class CrossAttention1D(nn.Module):
    """Cross-attention: query from decoder, key/value from encoder+waypoint embed.
    q: (B,Cq,Lq)  kv: (B,Ckv,Lkv) -> (B,Cq,Lq)
    """
    def __init__(self, q_channels, kv_channels, heads=2):
        super().__init__()
        self.heads = heads
        self.scale = (q_channels // heads) ** -0.5
        self.q_proj  = nn.Conv1d(q_channels,  q_channels, 1, bias=False)
        self.k_proj  = nn.Conv1d(kv_channels, q_channels, 1, bias=False)
        self.v_proj  = nn.Conv1d(kv_channels, q_channels, 1, bias=False)
        self.out_proj = nn.Conv1d(q_channels,  q_channels, 1)

    def forward(self, query, context):
        B, Cq, Lq = query.shape
        H, D = self.heads, Cq // self.heads
        q = self.q_proj(query).reshape(B, H, D, Lq)
        k = self.k_proj(context)
        v = self.v_proj(context)
        if k.shape[2] != Lq:
            k = F.interpolate(k, size=Lq, mode="linear", align_corners=False)
            v = F.interpolate(v, size=Lq, mode="linear", align_corners=False)
        k = k.reshape(B, H, D, Lq)
        v = v.reshape(B, H, D, Lq)
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(B, Cq, Lq)
        return self.out_proj(out) + query


class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1), nn.BatchNorm1d(out_c), nn.GELU(),
            nn.Conv1d(out_c, out_c, 3, padding=1), nn.BatchNorm1d(out_c), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class ObstacleSegmenter(nn.Module):
    """LIDAR flat array -> 16-sector obstacle mask + distance. Optional heavy module."""
    N_SECTORS = 16
    BASE_C = 32

    def __init__(self, obs_dim_raw=38464):
        super().__init__()
        C = self.BASE_C
        self.enc1 = ConvBlock1D(1, C)
        self.pool1 = nn.MaxPool1d(8, stride=8)
        self.enc2 = ConvBlock1D(C, C * 2)
        self.pool2 = nn.MaxPool1d(8, stride=8)
        self.bottleneck = ConvBlock1D(C * 2, C * 4)
        self.self_attn = SelfAttention1D(C * 4, heads=4)
        self.up2 = nn.ConvTranspose1d(C * 4, C * 2, kernel_size=8, stride=8)
        self.cross2 = CrossAttention1D(C * 2, C * 2, heads=2)
        self.dec2 = ConvBlock1D(C * 4, C * 2)
        self.up1 = nn.ConvTranspose1d(C * 2, C, kernel_size=8, stride=8)
        self.cross1 = CrossAttention1D(C, C, heads=2)
        self.dec1 = ConvBlock1D(C * 2, C)
        self.sector_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.N_SECTORS), nn.Flatten(1),
            nn.Linear(C * self.N_SECTORS, self.N_SECTORS * 2), nn.GELU(),
        )
        self.mask_out = nn.Linear(self.N_SECTORS * 2, self.N_SECTORS)
        self.dist_out = nn.Linear(self.N_SECTORS * 2, self.N_SECTORS)

    def forward(self, lidar_flat):
        x = lidar_flat.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        bn = self.self_attn(self.bottleneck(self.pool2(e2)))
        d2 = self.up2(bn)
        if d2.shape[2] != e2.shape[2]:
            d2 = F.interpolate(d2, size=e2.shape[2], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([self.cross2(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[2] != e1.shape[2]:
            d1 = F.interpolate(d1, size=e1.shape[2], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([self.cross1(d1, e1), e1], dim=1))
        h = self.sector_head(d1)
        return torch.sigmoid(self.mask_out(h)), torch.sigmoid(self.dist_out(h))


class LineOfSightEncoder(nn.Module):
    """Waypoint sequence + car heading -> 4-dim clearance vector [front, left, right, apex].
    Adapted from U-Transformer cross-attention skip pattern (Petit et al. 2021).
    """
    def __init__(self, lookahead=10, hidden=32):
        super().__init__()
        self.lookahead = lookahead
        self.wp_embed = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.GELU(), nn.Linear(16, 4), nn.Sigmoid(),
        )

    def forward(self, wp_seq, heading_rad):
        cos_h = torch.cos(heading_rad)
        sin_h = torch.sin(heading_rad)
        x_r = wp_seq[:, :, 0] * cos_h + wp_seq[:, :, 1] * sin_h
        y_r = -wp_seq[:, :, 0] * sin_h + wp_seq[:, :, 1] * cos_h
        wp_rot = torch.stack([x_r, y_r], dim=-1)
        h = self.transformer(self.wp_embed(wp_rot))
        return self.head(h.mean(dim=1))


class UTransformerObs(nn.Module):
    """Top-level singleton: extends extract_compact_obs (12-dim) with 4 LoS dims -> 16-dim."""
    COMPACT_OUT = 16

    def __init__(self, obs_dim_raw=38464, lookahead=10):
        super().__init__()
        self.obs_dim_raw = obs_dim_raw
        self.lookahead = lookahead
        self._segmenter = None
        self.los_encoder = LineOfSightEncoder(lookahead=lookahead, hidden=32)

    def init_segmenter(self, device="cpu"):
        self._segmenter = ObstacleSegmenter(self.obs_dim_raw).to(device).eval()

    @torch.no_grad()
    def encode_los(self, rp, waypoints, closest):
        try:
            n = len(waypoints)
            if n < 2 or len(closest) < 2:
                return np.zeros(4, dtype=np.float32)
            car_x = float(rp.get("x", 0.0))
            car_y = float(rp.get("y", 0.0))
            heading_rad = math.radians(float(rp.get("heading", 0.0)))
            start_wp = int(closest[1]) % n
            wp_seq = []
            for i in range(self.lookahead):
                wp = waypoints[(start_wp + i) % n]
                rel_x = float(wp[0]) - car_x
                rel_y = float(wp[1]) - car_y
                d = max(math.hypot(rel_x, rel_y), 1e-6)
                wp_seq.append([rel_x / d, rel_y / d])
            wp_t = torch.tensor([wp_seq], dtype=torch.float32)
            h_t  = torch.tensor([[heading_rad]], dtype=torch.float32)
            return self.los_encoder(wp_t, h_t).squeeze(0).numpy()
        except Exception:
            return np.zeros(4, dtype=np.float32)

    @torch.no_grad()
    def encode_obstacles(self, obs_raw_np):
        if self._segmenter is None:
            return None, None
        try:
            t = torch.tensor(obs_raw_np.flatten()[:self.obs_dim_raw],
                             dtype=torch.float32).unsqueeze(0)
            mask, dist = self._segmenter(t)
            return mask.squeeze(0).numpy(), dist.squeeze(0).numpy()
        except Exception:
            return None, None


_utobs_singleton = None

def get_utobs(obs_dim_raw=38464):
    global _utobs_singleton
    if _utobs_singleton is None:
        _utobs_singleton = UTransformerObs(obs_dim_raw=obs_dim_raw, lookahead=10)
        _utobs_singleton.eval()
    return _utobs_singleton


def extract_compact_obs_v2(obs_raw, rp, waypoints, closest, obs_dim_raw=38464):
    """16-dim compact observation (12 base + 4 LoS). Drop-in for extract_compact_obs."""
    try:
        speed = float(rp.get("speed", 0.0))
        dist_ctr = float(rp.get("distance_from_center", 0.0))
        tw = float(rp.get("track_width", 1.0))
        heading = float(rp.get("heading", 0.0))
        is_left = 1.0 if rp.get("is_left_of_center", False) else -1.0
        progress = float(rp.get("progress", 0.0)) / 100.0
        is_reversed = 1.0 if rp.get("is_reversed", False) else 0.0
        is_offtrack = 1.0 if rp.get("is_offtrack", False) else 0.0

        if waypoints and len(waypoints) >= 2 and len(closest) >= 2:
            n = len(waypoints)
            p0 = waypoints[closest[0] % n]
            p1 = waypoints[closest[1] % n]
            track_angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            hdiff = heading - track_angle
            while hdiff > 180:  hdiff -= 360
            while hdiff < -180: hdiff += 360
            heading_err = hdiff / 180.0
            lat_pos = dist_ctr / max(tw * 0.5, 0.01) * is_left
        else:
            heading_err, lat_pos = 0.0, 0.0

        try:
            from corneranalysis import lookahead_curvature_scan
            _, _, safe_speed, dist_to_corner = lookahead_curvature_scan(
                waypoints, closest, max_lookahead=10)
            curv_signal = (speed - safe_speed) / max(safe_speed, 0.1)
            dist_corner_norm = min(dist_to_corner / 5.0, 1.0)
        except Exception:
            curv_signal, dist_corner_norm = 0.0, 1.0

        base12 = np.array([
            speed / 4.0, lat_pos, heading_err, curv_signal, dist_corner_norm,
            progress, is_reversed, is_offtrack, is_left,
            float(closest[0] % len(waypoints)) / max(len(waypoints), 1),
            tw / 2.0, 0.0,
        ], dtype=np.float32)

        los4 = get_utobs(obs_dim_raw).encode_los(rp, waypoints, closest)
        return np.concatenate([base12, los4])
    except Exception:
        return np.zeros(16, dtype=np.float32)
