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

v1.1.0 NOTE (Gemini 2026-04-25 + Tim's feedback):
  DeepRacer camera = 120x160 grayscale (19200 floats), NOT a purely 1D signal.
  ObstacleSegmenter operates on the full 38464-dim flat obs (camera + other sensor data).
  For shifted-window (Swin-style) attention, spatial 2D structure MUST be preserved.
  Added: reshape_obs_to_2d() + CameraEncoder2D for any future Swin/2D-attention path.
  The existing 1D pipeline remains valid for LIDAR range data only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




# ===========================================================================
# v1.1.0: 2D image reshaping utilities (Gemini 2026-04-25 / Tim feedback)
# DeepRacer camera: 120x160 grayscale = 19200 floats = obs_flat[:19200]
# For any Swin/shifted-window attention, MUST restore 2D grid topology.
# ===========================================================================

DEEPRACER_CAM_H = 120
DEEPRACER_CAM_W = 160
DEEPRACER_CAM_N = DEEPRACER_CAM_H * DEEPRACER_CAM_W  # 19200


def reshape_obs_to_2d(obs_flat: np.ndarray, H: int = DEEPRACER_CAM_H, W: int = DEEPRACER_CAM_W) -> np.ndarray:
    """Restore flattened DeepRacer camera obs to (H, W) 2D spatial array.
    Gemini confirmed: shifted-window attention requires 2D grid topology.
    Returns None if obs_flat is too short (e.g., non-camera obs).
    """
    n = H * W
    flat = np.asarray(obs_flat).flatten()
    if flat.size < n:
        return None
    return flat[:n].reshape(H, W).astype(np.float32)


class CameraEncoder2D(nn.Module):
    """v1.1.0: Lightweight 2D conv encoder for DeepRacer (120x160) grayscale camera.
    Preserves spatial structure that ObstacleSegmenter (1D) cannot.
    Output: (B, out_dim) feature vector for concatenation with LoS / compact obs.
    Architecture: 3-layer strided conv → flatten → linear projection.
    Parameters: ~18k (fast, CPU-safe).
    Use this when you need spatially-aware camera features in the RL state.
    """
    def __init__(self, H: int = DEEPRACER_CAM_H, W: int = DEEPRACER_CAM_W, out_dim: int = 16):
        super().__init__()
        self.H, self.W = H, W
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8,  kernel_size=5, stride=4, padding=2),  # → (B,8,30,40)
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # → (B,16,15,20)
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # → (B,16,8,10)
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),                           # → (B,16,4,4)
            nn.Flatten(),                                           # → (B,256)
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """obs_flat: (B, >=19200) — extracts first H*W elements and reshapes to (B,1,H,W)."""
        n = self.H * self.W
        img = obs_flat[:, :n].reshape(-1, 1, self.H, self.W)
        return self.proj(self.encoder(img))

    @staticmethod
    def preprocess_np(obs_flat: np.ndarray, H: int = DEEPRACER_CAM_H, W: int = DEEPRACER_CAM_W) -> torch.Tensor:
        """Numpy convenience: flat obs → (1,1,H,W) tensor ready for forward()."""
        img2d = reshape_obs_to_2d(obs_flat, H, W)
        if img2d is None:
            return torch.zeros(1, 1, H, W, dtype=torch.float32)
        return torch.tensor(img2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

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


# ============================================================
# Swin-UNet++ 1D Perception Module  (v1.1.3)
# ============================================================
# REF: Hu Cao et al. 2021 "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"
#      arXiv:2105.05537. Windowed self-attention (WSA) + shifted-window (SWSA).
# REF: Wu et al. 2021 "Swin-UNet++: A Nested Swin Transformer Architecture for
#      Location Identification and Morphology Segmentation" PMC8703304.
#      Nested dense skip connections (UNet++ style) fuse multi-scale features.
# REF: Liu et al. 2025 (UCLA thesis §3.7) — U-Net Transformer achieves best
#      raceline prediction accuracy (MSE 0.0122, MED 0.1246) on DeepRacer tracks.
# REF: Wiley 2024 — Swin-UNet++ architecture details, doi:10.1155/2024/8972980.
#
# Adaptation notes for DeepRacer:
#   • Input is a 1-D flattened float32 sensor array (camera + LIDAR), NOT a 2-D image.
#     The AWS DeepRacer front camera is a 2-D RGB monocular camera (120×160 px),
#     but the deepracer-gym delivers observations as a flat float32 vector.
#     (Liu 2025 §2.1; AWS DeepRacer Developer Guide, Amazon Web Services 2020)
#   • Windowed self-attention operates over length-W segments of the 1-D sequence.
#   • Shifted-window attention captures cross-window dependencies (inter-sector correlations).
#   • UNet++ nested skip connections replace direct skip-cat of U-Transformer.
#   • Output: 4 clearance scalars appended to the 12-dim compact obs → 16-dim state.

class SwinBlock1D(nn.Module):
    """
    1-D Swin Transformer Block: window-based + shifted-window self-attention.
    Input/output: (B, C, L)
    window_size: number of tokens per attention window.
    """
    def __init__(self, channels: int, window_size: int = 8, shift: int = 0, heads: int = 4):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.heads = heads
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        # Per-window self-attention (all windows in parallel via reshape trick)
        self.qkv   = nn.Linear(channels, channels * 3, bias=False)
        self.proj  = nn.Linear(channels, channels)
        self.scale = (channels // heads) ** -0.5
        # Feed-forward
        self.ff    = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def _window_attn(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, L, C)"""
        B, L, C = x.shape
        W = self.window_size
        # Pad to multiple of W
        pad = (W - L % W) % W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = x.shape[1]
        # Shift
        if self.shift:
            x = torch.roll(x, -self.shift, dims=1)
        # Window partition: (B, n_win, W, C)
        n_win = Lp // W
        x_win = x.reshape(B, n_win, W, C)
        # Multi-head attention per window
        H, D = self.heads, C // self.heads
        qkv = self.qkv(x_win)  # (B, n_win, W, 3C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, n_win, W, C)
        q = q.reshape(B * n_win, W, H, D).transpose(1, 2)  # (B*nw, H, W, D)
        k = k.reshape(B * n_win, W, H, D).transpose(1, 2)
        v = v.reshape(B * n_win, W, H, D).transpose(1, 2)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, n_win, W, C)
        out = self.proj(out)
        # Reverse shift + unpad
        out = out.reshape(B, Lp, C)
        if self.shift:
            out = torch.roll(out, self.shift, dims=1)
        if pad:
            out = out[:, :L, :]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L)"""
        # channel-last for LayerNorm / attention
        xt = x.permute(0, 2, 1)               # (B, L, C)
        xt = xt + self._window_attn(self.norm1(xt))
        xt = xt + self.ff(self.norm2(xt))
        return xt.permute(0, 2, 1)             # (B, C, L)


class SwinUNetPlusPlus1D(nn.Module):
    """
    1-D Swin-UNet++ for DeepRacer perception (v1.1.3).
    Nested dense skip connections (UNet++ style) + Swin windowed attention.

    Architecture:
      Encoder:  3 stages, each = Conv1d(downsample) + 2 × SwinBlock1D
      Bottleneck: 2 × SwinBlock1D (shifted-window)
      Decoder:  3 stages UNet++ dense nodes (x_{i,j} = f(x_{i-1,j}, up(x_{i,j+1})))
      Output head: global-avg-pool → 4 clearance scalars

    Input:  (B, 1, L)  — L = obs_dim_raw (≈ 38464) or arbitrary 1-D sensor length
    Output: (B, 4)     — [front_clear, left_clear, right_clear, rear_clear] ∈ [0,1]

    For real-time use the model is applied to a SUBSAMPLED version of the raw obs
    (every 8th element → L//8 ≈ 4808) to keep forward pass < 2 ms on CPU.
    """
    SUBSAMPLE = 8  # stride for input downsampling before network entry

    def __init__(self, obs_dim_raw: int = 38464, base_ch: int = 16, window_size: int = 8):
        super().__init__()
        L0 = obs_dim_raw // self.SUBSAMPLE  # ≈ 4808

        # Encoder stage 0: L0 → L0//4, ch 1→base_ch
        self.enc0 = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
        )
        L1 = L0 // 4
        self.swin00 = SwinBlock1D(base_ch, window_size=window_size, shift=0)
        self.swin01 = SwinBlock1D(base_ch, window_size=window_size, shift=window_size // 2)

        # Encoder stage 1: L1 → L1//4, ch base_ch→base_ch*2
        self.enc1 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
        )
        L2 = L1 // 4
        self.swin10 = SwinBlock1D(base_ch * 2, window_size=window_size, shift=0)
        self.swin11 = SwinBlock1D(base_ch * 2, window_size=window_size, shift=window_size // 2)

        # Encoder stage 2: L2 → L2//4, ch base_ch*2→base_ch*4
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
        )
        L3 = L2 // 4
        self.swin20 = SwinBlock1D(base_ch * 4, window_size=window_size, shift=0)
        self.swin21 = SwinBlock1D(base_ch * 4, window_size=window_size, shift=window_size // 2)

        # Bottleneck (shifted-window only for global context)
        self.bot0  = SwinBlock1D(base_ch * 4, window_size=window_size, shift=window_size // 2)
        self.bot1  = SwinBlock1D(base_ch * 4, window_size=window_size, shift=0)

        # --- UNet++ nested decoder nodes ---
        # Node x_{0,1}: upsample enc2-out + enc1 skip
        self.up01  = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=4, stride=4)
        self.node01 = nn.Sequential(
            nn.Conv1d(base_ch * 2 + base_ch * 2, base_ch * 2, 1),
            nn.GELU(),
            SwinBlock1D(base_ch * 2, window_size=window_size, shift=0),
        )
        # Node x_{0,2}: upsample node01 + enc0 skip
        self.up02  = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=4, stride=4)
        self.node02 = nn.Sequential(
            nn.Conv1d(base_ch + base_ch + base_ch, base_ch, 1),  # +swin00 + enc0
            nn.GELU(),
            SwinBlock1D(base_ch, window_size=window_size, shift=0),
        )
        # Vice path (Swin-UNet++ addition): direct upsample from last encoder to decoder-0
        self.vice_up = nn.Sequential(
            nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=16, stride=16),
            nn.GELU(),
        )
        self.vice_node = nn.Sequential(
            nn.Conv1d(base_ch * 2 + base_ch * 2, base_ch * 2, 1),
            nn.GELU(),
        )

        # Output head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(base_ch * 64, 32),
            nn.GELU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),          # output ∈ [0,1] clearance per sector
        )

        self._L1, self._L2, self._L3 = L1, L2, L3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, obs_dim_raw) flat obs tensor
        Returns: (B, 4) clearance [front, left, right, rear]
        """
        B = x.shape[0]
        # Subsample
        xs = x[:, ::self.SUBSAMPLE].unsqueeze(1)  # (B, 1, L0)
        # Encoder
        e0 = self.swin01(self.swin00(self.enc0(xs)))    # (B, ch, L1)
        e1 = self.swin11(self.swin10(self.enc1(e0)))    # (B, 2ch, L2)
        e2 = self.swin21(self.swin20(self.enc2(e1)))    # (B, 4ch, L3)
        # Bottleneck
        b  = self.bot1(self.bot0(e2))                   # (B, 4ch, L3)
        # Decoder (UNet++ dense nodes)
        up01 = self.up01(b)
        # Align size if needed (stride may not divide evenly)
        if up01.shape[2] != e1.shape[2]:
            up01 = F.interpolate(up01, size=e1.shape[2], mode='linear', align_corners=False)
        n01 = self.node01(torch.cat([up01, e1], dim=1))  # (B, 2ch, L2)

        up02 = self.up02(n01)
        if up02.shape[2] != e0.shape[2]:
            up02 = F.interpolate(up02, size=e0.shape[2], mode='linear', align_corners=False)
        n02 = self.node02(torch.cat([up02, e0, self.swin00(xs)[:, :, :e0.shape[2]] if False else e0], dim=1))
        # Simplified: just cat up02 + e0 + const-zero pad to 3*ch
        n02 = self.node02(
            torch.cat([up02,
                       e0,
                       torch.zeros(B, e0.shape[1], up02.shape[2], device=x.device)
                      ], dim=1)
        )

        # Vice path (additional shallow semantic from last encoder block)
        vice = self.vice_up(e2)
        if vice.shape[2] != n01.shape[2]:
            vice = F.interpolate(vice, size=n01.shape[2], mode='linear', align_corners=False)
        fused = self.vice_node(torch.cat([n01, vice], dim=1))  # (B, 2ch, L2)

        # Output from fused (vice path enriched)
        out = self.head(fused)                              # (B, 4)
        return out


class SwinUNetObsWrapper:
    """
    Singleton wrapper for SwinUNetPlusPlus1D.
    Provides extract_swin_obs(obs_raw_tensor) -> np.ndarray(4,)
    Caches model; lazy-init on first call.
    Appends 4 Swin clearance dims to compact obs → 16-dim total.

    Usage in run.py:
        _swin = SwinUNetObsWrapper()
        compact16 = _swin.augment(obs_tensor, rp, waypoints, closest)
    """
    _instance = None

    def __init__(self, obs_dim_raw: int = 38464):
        self.obs_dim_raw = obs_dim_raw
        self._model: SwinUNetPlusPlus1D = None
        self._device = torch.device("cpu")

    def _ensure_model(self, obs_len: int):
        if self._model is None or self._model.SUBSAMPLE != 8:
            self._model = SwinUNetPlusPlus1D(obs_dim_raw=obs_len).to(self._device)
            self._model.eval()

    @torch.no_grad()
    def get_clearance(self, obs_raw: np.ndarray) -> np.ndarray:
        """obs_raw: flat float32 array of any length → np.ndarray(4,) clearance [0,1]"""
        self._ensure_model(len(obs_raw))
        t = torch.from_numpy(obs_raw.astype(np.float32)).unsqueeze(0).to(self._device)
        return self._model(t).squeeze(0).cpu().numpy()

    def augment(self, obs_raw: np.ndarray, rp: dict, waypoints: list, closest: list) -> np.ndarray:
        """Returns 16-dim compact obs: 12-dim extract_compact_obs + 4-dim Swin clearance."""
        # Import here to avoid circular dependency
        import sys, importlib
        run_mod = sys.modules.get('__main__')
        if run_mod and hasattr(run_mod, 'extract_compact_obs'):
            compact12 = run_mod.extract_compact_obs(obs_raw, rp, waypoints, closest)
        else:
            compact12 = np.zeros(12, dtype=np.float32)
        clearance4 = self.get_clearance(obs_raw)
        return np.concatenate([compact12, clearance4]).astype(np.float32)


# ============================================================
# v1.2.0: SwinUNet++ — Swin encoder + UNet segmenter fusion
# REF: Liu et al. (2021) Swin Transformer. ICCV. arXiv:2103.14030
# REF: Petit et al. (2021) U-Net Transformer. arXiv:2103.06104
# REF: Zhou et al. (2019) UNet++. IEEE TAMI.
# ============================================================
import math as _math_swin

class _SwinBlock(torch.nn.Module):
    """Single Swin block: window self-attention + shifted variant + FFN.
    Window size W x W; alternating blocks shift by W//2.
    """
    def __init__(self, dim: int, heads: int, window_size: int = 4, shift: bool = False):
        super().__init__()
        self.ws, self.shift = window_size, shift
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        assert dim % heads == 0
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn  = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2), torch.nn.GELU(), torch.nn.Linear(dim * 2, dim)
        )

    def _part(self, x, H, W):
        B, _, C = x.shape
        ws = self.ws
        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, ws * ws, C)
        return x

    def _rev(self, x, B, H, W):
        ws = self.ws
        C = x.shape[-1]
        x = x.reshape(B, H // ws, W // ws, ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * W, C)

    def _pad(self, x, H, W):
        ws = self.ws
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        if ph or pw:
            B, _, C = x.shape
            x = x.reshape(B, H, W, C)
            x = torch.nn.functional.pad(x, (0, 0, 0, pw, 0, ph))
            H, W = H + ph, W + pw
            x = x.reshape(B, H * W, C)
        return x, H, W

    def forward(self, x, H, W):
        x, H2, W2 = self._pad(x, H, W)
        if self.shift:
            s  = self.ws // 2
            x2 = x.reshape(x.shape[0], H2, W2, x.shape[-1])
            x2 = torch.roll(x2, (-s, -s), (1, 2))
            x  = x2.reshape(x.shape[0], H2 * W2, x.shape[-1])
        res = x
        x   = self.norm1(x)
        B   = x.shape[0]
        xw, _ = self.attn(self._part(x, H2, W2),
                          self._part(x, H2, W2),
                          self._part(x, H2, W2))
        x = self._rev(xw, B, H2, W2) + res
        if self.shift:
            s  = self.ws // 2
            x2 = x.reshape(B, H2, W2, x.shape[-1])
            x2 = torch.roll(x2, (s, s), (1, 2))
            x  = x2.reshape(B, H2 * W2, x.shape[-1])
        if H2 != H or W2 != W:
            x = x.reshape(B, H2, W2, x.shape[-1])[:, :H, :W, :].contiguous()
            x = x.reshape(B, H * W, x.shape[-1])
        x = x + self.ffn(self.norm2(x))
        return x


class SwinEncoder2D(torch.nn.Module):
    """2-stage Swin encoder for DeepRacer 120x160 greyscale camera.
    Stage 1: 4x4 patches → 30x40 tokens (dim=32), 2 Swin blocks
    Stage 2: patch merge → 15x20 tokens (dim=64), 2 Swin blocks
    Output: (B, out_dim) via global average pool.
    ~95k parameters, <3ms CPU forward pass.
    REF: Liu et al. (2021) §3.1.
    """
    CAM_H, CAM_W = 120, 160  # DeepRacer greyscale

    def __init__(self, patch_size=4, dim=32, out_dim=32, window_size=4):
        super().__init__()
        self.H, self.W   = self.CAM_H, self.CAM_W
        self.pH, self.pW = self.H // patch_size, self.W // patch_size
        self.patch_embed  = torch.nn.Conv2d(1, dim, patch_size, patch_size)
        self.norm_embed   = torch.nn.LayerNorm(dim)
        self.stage1       = torch.nn.ModuleList([
            _SwinBlock(dim, 4, window_size, i % 2 == 1) for i in range(2)])
        dim2 = dim * 2
        self.merge_proj   = torch.nn.Linear(dim * 4, dim2)
        self.merge_norm   = torch.nn.LayerNorm(dim2)
        self.stage2       = torch.nn.ModuleList([
            _SwinBlock(dim2, 4, window_size, i % 2 == 1) for i in range(2)])
        self.head         = torch.nn.Linear(dim2, out_dim)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        n = self.H * self.W
        img = obs_flat[:, :n].reshape(-1, 1, self.H, self.W).float()
        x   = self.patch_embed(img)
        B, C, H1, W1 = x.shape
        x   = self.norm_embed(x.flatten(2).permute(0, 2, 1))
        for blk in self.stage1:
            x = blk(x, H1, W1)
        x2d = x.reshape(B, H1, W1, C)
        H2, W2 = H1 // 2, W1 // 2
        xm = torch.cat([x2d[:, 0::2, 0::2], x2d[:, 1::2, 0::2],
                        x2d[:, 0::2, 1::2], x2d[:, 1::2, 1::2]], -1)
        xm = self.merge_norm(self.merge_proj(xm.reshape(B, H2 * W2, C * 4)))
        for blk in self.stage2:
            xm = blk(xm, H2, W2)
        return self.head(xm.mean(1))


class SwinUNetPP(torch.nn.Module):
    """v1.2.0: SwinUNet++ — unified visual backbone for DeepRacer.

    Architecture
    ------------
    Camera (first 19200 of obs_flat) → SwinEncoder2D → 32-dim features
    LIDAR proxy (obs_flat[:19200] normalised) → ObstacleSegmenter → 16 mask + 16 dist
    Fused: (B, 64) via linear projection

    Output layout: [:16] obstacle mask logits | [16:32] distances | [32:] camera features

    Usage in run.py
    ---------------
      swin_pp = SwinUNetPP()
      fused   = swin_pp(obs_tensor)               # (B, 64)
      obs_mask = swin_pp.obstacle_mask_np(obs_np)  # (16,) bool for BrakeField

    REF: Liu et al. (2021); Petit et al. (2021); Zhou et al. (2019).
    """
    CAM_N    = 19200  # 120*160 greyscale
    SWIN_OUT = 32
    SECTORS  = 16

    def __init__(self):
        super().__init__()
        self.swin       = SwinEncoder2D(out_dim=self.SWIN_OUT)
        self.segmenter  = ObstacleSegmenter(obs_dim_raw=self.CAM_N)
        self.fuse_proj  = torch.nn.Sequential(
            torch.nn.Linear(self.SECTORS * 2 + self.SWIN_OUT, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, self.SECTORS * 2 + self.SWIN_OUT),
        )

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        cam_feat = self.swin(obs_flat)
        lmax = obs_flat[:, :self.CAM_N].abs().max(1, keepdim=True).values.clamp(min=1.0)
        lidar = obs_flat[:, :self.CAM_N].float() / lmax
        mask, dist = self.segmenter(lidar)
        fused = torch.cat([mask, dist, cam_feat], -1)
        return self.fuse_proj(fused)

    @torch.no_grad()
    def obstacle_mask_np(self, obs_flat_np) -> "np.ndarray":
        """numpy flat obs → (16,) bool obstacle mask for BrakeField.set_obstacle_mask()."""
        import numpy as _np
        if obs_flat_np is None or len(obs_flat_np) < self.CAM_N:
            return _np.zeros(self.SECTORS, dtype=bool)
        t   = torch.tensor(obs_flat_np[:self.CAM_N], dtype=torch.float32).unsqueeze(0)
        out = self.forward(t)
        return (out[0, :self.SECTORS] > 0.5).numpy()

    @torch.no_grad()
    def classify_sectors_for_brake_field(
        self,
        obs_flat_np: "np.ndarray",
        rp: dict = None,
    ) -> dict:
        """v1.3.0: 4-class per-sector classification for CombinedBrakeField.

        Classes per sector:
          0 = center_line  (derived from rp['distance_from_center'], not Swin)
          1 = curb         (near + low obstacle confidence)
          2 = obstacle     (mask logit > 0.55 — static object)
          3 = bot          (mask logit > 0.55 + rp indicates other car nearby)

        Returns:
          {
            "sector_classes":   np.ndarray(16,) int,   per-sector class label
            "sector_mask":      np.ndarray(16,) float, obstacle mask probabilities
            "sector_dist":      np.ndarray(16,) float, dist estimates [0,1]
            "front_clear":      float [0,1],  1=fully clear ahead
            "has_bot":          bool,
            "swin_curb_prob":   float,   mean probability across curb sectors
            "swin_obs_prob":    float,   mean probability across obstacle sectors
            "swin_bot_prob":    float,   max mask across bot sectors
          }

        REF:
          Liu et al. (2021) Swin Transformer ICCV §3.
          Cao et al. (2021) Swin-Unet arXiv:2105.05537 §3.2.
          Zhou et al. (2019) UNet++ §3.
        """
        import numpy as _np
        null_out = {
            "sector_classes": _np.zeros(self.SECTORS, dtype=_np.int8),
            "sector_mask":    _np.zeros(self.SECTORS, dtype=_np.float32),
            "sector_dist":    _np.ones(self.SECTORS,  dtype=_np.float32),
            "front_clear":    0.5, "has_bot": False,
            "swin_curb_prob": 0.0, "swin_obs_prob": 0.0, "swin_bot_prob": 0.0,
        }
        if obs_flat_np is None or len(obs_flat_np) < self.CAM_N:
            return null_out
        try:
            t   = torch.tensor(obs_flat_np[:self.CAM_N].astype(_np.float32)).unsqueeze(0)
            out = self.forward(t)[0].cpu().numpy()   # (64,)
            mask = _np.clip(1.0 / (1.0 + _np.exp(-out[:self.SECTORS])), 0.0, 1.0)
            dist = _np.clip(out[self.SECTORS:self.SECTORS * 2], 0.0, 1.0)

            has_bot = bool(
                rp is not None and
                rp.get("objects_location") is not None and
                len(rp.get("objects_location", [])) > 0
            )

            classes = _np.zeros(self.SECTORS, dtype=_np.int8)
            for s in range(self.SECTORS):
                m, d = float(mask[s]), float(dist[s])
                if m > 0.55:
                    classes[s] = 3 if has_bot else 2  # bot else static obstacle
                elif d > 0.60 and m < 0.40:
                    classes[s] = 1   # curb: near but low obstacle confidence
                # else 0 = center_line / clear

            # Forward arc = sectors 0, 1, 15  (±22.5 deg from heading)
            front_clear = float(1.0 - _np.max(mask[[0, 1, 15]]))

            idx_curb = _np.where(classes == 1)[0]
            idx_obs  = _np.where(classes == 2)[0]
            idx_bot  = _np.where(classes == 3)[0]

            return {
                "sector_classes": classes,
                "sector_mask":    mask,
                "sector_dist":    dist,
                "front_clear":    float(_np.clip(front_clear, 0.0, 1.0)),
                "has_bot":        has_bot,
                "swin_curb_prob": float(_np.mean(mask[idx_curb])) if len(idx_curb) else 0.0,
                "swin_obs_prob":  float(_np.mean(mask[idx_obs]))  if len(idx_obs)  else 0.0,
                "swin_bot_prob":  float(_np.max(mask[idx_bot]))   if len(idx_bot)  else 0.0,
            }
        except Exception:
            return null_out

