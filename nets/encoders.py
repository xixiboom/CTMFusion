from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ConvGNReLU, ResidualBlock

try:
    import pytorch_wavelets as pw
    _HAS_PYTORCH_WAVELETS = True
except Exception:
    pw = None
    _HAS_PYTORCH_WAVELETS = False


class FrequencyFrontEnd(nn.Module):
    def __init__(self, in_channels: int = 3, J: int = 2, backend: str = "auto"):
        super().__init__()
        self.in_channels = in_channels
        self.J = J
        self.backend = backend.lower()
        if self.backend not in {"auto", "dtcwt", "fallback"}:
            raise ValueError("backend must be one of: auto, dtcwt, fallback")

        self.use_dtcwt = self.backend in {"auto", "dtcwt"} and _HAS_PYTORCH_WAVELETS
        if self.backend == "dtcwt" and not _HAS_PYTORCH_WAVELETS:
            raise ImportError("pytorch_wavelets is required for backend='dtcwt'.")

        self.dtcwt = pw.DTCWTForward(J=J, biort="near_sym_b", qshift="qshift_b") if self.use_dtcwt else None

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap", lap)

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2 ** self.J
        h, w = x.shape[-2:]
        if h < factor or w < factor:
            return F.interpolate(x, size=(max(1, h // 2), max(1, w // 2)), mode="bilinear", align_corners=False)
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)

    def _fallback_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .base import _rgb_to_gray
        low = self._lowpass(x)
        target_size = low.shape[2:]
        gray = _rgb_to_gray(x)
        feats = []
        energy = 0.0
        for j in range(self.J):
            scale = 2 ** j
            g = F.avg_pool2d(gray, kernel_size=scale, stride=scale) if scale > 1 else gray
            gx = F.conv2d(g, self.sobel_x, padding=1)
            gy = F.conv2d(g, self.sobel_y, padding=1)
            lap = F.conv2d(g, self.lap, padding=1)
            grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
            mean = F.avg_pool2d(g, kernel_size=5, stride=1, padding=2)
            var = F.avg_pool2d((g - mean).pow(2), kernel_size=5, stride=1, padding=2)
            high = torch.abs(g - mean)
            bands = torch.cat([gx.abs(), gy.abs(), lap.abs(), grad, var, high], dim=1)
            bands = F.adaptive_avg_pool2d(bands, target_size)
            feats.append(bands)
            energy = energy + bands[:, 3:4] + 0.25 * bands[:, 2:3] + 0.25 * bands[:, 4:5]
        high_cat = torch.cat(feats, dim=1)
        energy = energy / float(self.J)
        energy = energy / (F.avg_pool2d(energy, kernel_size=9, stride=1, padding=4) + 1e-6)
        return low, high_cat, energy

    def _dtcwt_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        yL, yH = self.dtcwt(x)
        target_size = yL.shape[2:]
        high_mags = []
        energy = 0.0
        for scale_bands in yH:
            mag = torch.abs(torch.view_as_complex(scale_bands.contiguous())).mean(dim=1)
            if mag.shape[2:] != target_size:
                mag = F.adaptive_avg_pool2d(mag, target_size)
            high_mags.append(mag)
            energy = energy + mag.mean(dim=1, keepdim=True)
        high_cat = torch.cat(high_mags, dim=1)
        energy = energy / float(len(yH))
        energy = energy / (F.avg_pool2d(energy, kernel_size=9, stride=1, padding=4) + 1e-6)
        return yL, high_cat, energy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_dtcwt:
            return self._dtcwt_forward(x)
        return self._fallback_forward(x)


class WaveletEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_dim: int = 32, J: int = 2, backend: str = "auto"):
        super().__init__()
        self.J = J
        self.freq = FrequencyFrontEnd(in_channels=in_channels, J=J, backend=backend)
        self.low_proj = ConvGNReLU(in_channels, base_dim, num_groups=4)
        self.stem = nn.Sequential(ConvGNReLU(base_dim, base_dim, num_groups=4), ResidualBlock(base_dim, num_groups=4))
        self.stage1 = nn.Sequential(
            ConvGNReLU(base_dim, base_dim * 2, stride=2, num_groups=8),
            ResidualBlock(base_dim * 2, num_groups=8),
        )
        self.stage2 = nn.Sequential(
            ConvGNReLU(base_dim * 2, base_dim * 4, stride=2, num_groups=16),
            ResidualBlock(base_dim * 4, num_groups=16),
        )
        self.stage3 = nn.Sequential(
            ConvGNReLU(base_dim * 4, base_dim * 4, stride=2, num_groups=16),
            ResidualBlock(base_dim * 4, num_groups=16),
        )
        self.prior_head = nn.Sequential(
            nn.Conv2d(6 * J, base_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_valid_num_groups(4, base_dim), base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from .base import _valid_num_groups
        low, high_cat, energy = self.freq(x)
        low_feat = self.low_proj(low)
        x0 = self.stem(low_feat)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        learned_prior = self.prior_head(high_cat)
        return x1, x2, x3, learned_prior, energy


class FullResolutionDetailEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNReLU(in_channels, dim, num_groups=4),
            ResidualBlock(dim, num_groups=4),
            ResidualBlock(dim, num_groups=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
