from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _safe_logit, _rgb_to_gray, ConvGNReLU, ResidualBlock


class FocusPriorEstimator(nn.Module):
    def __init__(self, temp: float = 4.0):
        super().__init__()
        self.temp = temp
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap", lap)

    def focus_energy(self, x: torch.Tensor) -> torch.Tensor:
        gray = _rgb_to_gray(x)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        lap = F.conv2d(gray, self.lap, padding=1)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        mean5 = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        var5 = F.avg_pool2d((gray - mean5).pow(2), kernel_size=5, stride=1, padding=2)
        high = torch.abs(gray - mean5)

        energy = grad + 0.30 * lap.abs() + 0.45 * torch.sqrt(var5 + 1e-6) + 0.25 * high

        e5 = F.avg_pool2d(energy, kernel_size=5, stride=1, padding=2)
        e9 = F.avg_pool2d(energy, kernel_size=9, stride=1, padding=4)
        e15 = F.avg_pool2d(energy, kernel_size=15, stride=1, padding=7)
        energy = 0.50 * e5 + 0.30 * e9 + 0.20 * e15

        local_mean = F.avg_pool2d(energy, kernel_size=31, stride=1, padding=15)
        return energy / (local_mean + 1e-6)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eA = self.focus_energy(A)
        eB = self.focus_energy(B)
        signed_focus = (eA - eB) / (eA + eB + 1e-6)
        signed_focus = signed_focus.clamp(-1.0, 1.0)

        confidence = F.avg_pool2d(signed_focus.abs(), kernel_size=7, stride=1, padding=3).clamp(0.0, 1.0)
        prior_decision = torch.sigmoid(self.temp * signed_focus).clamp(1e-4, 1.0 - 1e-4)
        return prior_decision, confidence, signed_focus, eA, eB


class ProgressiveRegionDecoder(nn.Module):
    def __init__(self, dim_deep: int = 128, skip_dims: Tuple[int, int] = (64, 128), dropout: float = 0.0):
        super().__init__()
        from .base import _valid_num_groups
        d1, d2 = skip_dims
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_deep, d2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(_valid_num_groups(8, d2), d2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(d2, d1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(_valid_num_groups(8, d1), d1),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(d1, 32 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(dropout)
        self.final = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 3, padding=1))
        nn.init.normal_(self.final[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.final[-1].bias)

    @staticmethod
    def _align_size(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if src.shape[2:] == target.shape[2:]:
            return src
        return F.interpolate(src, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, deep_feat: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        x = self.up1(deep_feat)
        x = self._align_size(x, skip2) + skip2
        x = self.up2(x)
        x = self._align_size(x, skip1) + skip1
        x = self.up3(x)
        x = self.drop(x)
        return self.final(x)


class FullResolutionDecisionHead(nn.Module):
    def __init__(self, img_channels: int = 3, detail_dim: int = 32, hidden: int = 32):
        super().__init__()
        in_ch = 3 * img_channels + 3 * detail_dim + 5
        self.body = nn.Sequential(
            ConvGNReLU(in_ch, hidden, num_groups=8),
            ResidualBlock(hidden, num_groups=8),
            ConvGNReLU(hidden, hidden, num_groups=8),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.body[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, A: torch.Tensor, B: torch.Tensor,
                feat_A: torch.Tensor, feat_B: torch.Tensor,
                prior_decision: torch.Tensor, confidence: torch.Tensor,
                signed_focus: torch.Tensor, ctm_decision: torch.Tensor,
                sync_map: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            A, B, torch.abs(A - B),
            feat_A, feat_B, torch.abs(feat_A - feat_B),
            prior_decision, confidence, signed_focus, ctm_decision, sync_map,
        ], dim=1)
        return self.body(x)


class EdgeAnchoredDecisionRefiner(nn.Module):
    def __init__(self, inner_iter: int = 4, init_k: float = 0.16,
                 init_alpha: float = 0.48, sharpen: float = 1.25,
                 boundary_sigma: float = 0.115, curvature: float = 0.18):
        super().__init__()
        self.inner_iter = inner_iter
        self.k_raw = nn.Parameter(torch.tensor(init_k).log())
        self.alpha_raw = nn.Parameter(torch.tensor(init_alpha).logit())
        self.sharpen_raw = nn.Parameter(torch.tensor(sharpen - 1.0))
        self.boundary_sigma = boundary_sigma
        self.curvature_raw = nn.Parameter(torch.tensor(curvature).log())

        kernels, weights = [], []
        coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dy, dx in coords:
            k = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
            k[0, 0, dy + 1, dx + 1] = 1.0
            kernels.append(k)
            weights.append(0.7071 if abs(dy) + abs(dx) == 2 else 1.0)
        self.register_buffer("neighbor_kernel", torch.cat(kernels, dim=0))
        self.register_buffer("neighbor_weight", torch.tensor(weights, dtype=torch.float32).view(1, 8, 1, 1))

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        gauss = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 16.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap_kernel", lap)
        self.register_buffer("gauss_kernel", gauss)

    @property
    def k(self) -> torch.Tensor:
        return F.softplus(self.k_raw) + 1e-4

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw) * 0.70 + 0.04

    @property
    def sharpen(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.sharpen_raw)

    @property
    def curvature_strength(self) -> torch.Tensor:
        return F.softplus(self.curvature_raw).clamp(0.02, 0.45)

    def _edge(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        g = 0.5 * _rgb_to_gray(A) + 0.5 * _rgb_to_gray(B)
        gx = F.conv2d(g, self.sobel_x, padding=1)
        gy = F.conv2d(g, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return (edge / (edge.mean(dim=[2, 3], keepdim=True) + 1e-6)).clamp(0.0, 4.0)

    def _boundary_band(self, D: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((D - 0.5).pow(2)) / (2.0 * self.boundary_sigma ** 2)).clamp(0.0, 1.0)

    def forward(self, decision: torch.Tensor, confidence: torch.Tensor, sync_map: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decision0 = decision.clamp(1e-4, 1.0 - 1e-4)
        conf = confidence.clamp(0.0, 1.0)
        sync = sync_map.clamp(0.0, 1.0)
        guide_gray = 0.5 * _rgb_to_gray(A) + 0.5 * _rgb_to_gray(B)
        image_edge = self._edge(A, B)
        edge_prob = torch.sigmoid(image_edge - 0.85)

        boundary0 = self._boundary_band(decision0)
        uncertainty = 4.0 * decision0 * (1.0 - decision0)

        interior = (1.0 - boundary0).clamp(0.0, 1.0)
        anchor = 0.18 + 0.56 * conf * (0.55 + 0.45 * interior) + 0.14 * sync * (0.65 + 0.35 * interior)
        anchor = anchor + 0.12 * edge_prob * (0.45 + 0.55 * interior)
        anchor = anchor * (1.0 - 0.16 * uncertainty)
        anchor = anchor.clamp(0.12, 0.88)

        current = decision0
        for _ in range(self.inner_iter):
            boundary = self._boundary_band(current)

            n_dec = F.conv2d(current, self.neighbor_kernel, padding=1)
            n_conf = F.conv2d(conf, self.neighbor_kernel, padding=1)
            n_sync = F.conv2d(sync, self.neighbor_kernel, padding=1)
            n_gray = F.conv2d(guide_gray, self.neighbor_kernel, padding=1)
            gray_diff = torch.abs(n_gray - guide_gray)
            edge_barrier = torch.exp(-gray_diff.pow(2) / self.k.pow(2))
            smooth_gain = 0.35 + 0.65 * boundary
            weight = edge_barrier * self.neighbor_weight * smooth_gain * (0.05 + 0.30 * n_conf + 0.18 * n_sync)
            smooth_target = (n_dec * weight).sum(dim=1, keepdim=True) / (weight.sum(dim=1, keepdim=True) + 1e-6)
            current = current + self.alpha * (smooth_target - current) * (0.35 + 0.65 * boundary)

            logit_current = _safe_logit(current)
            lap_logit = F.conv2d(logit_current, self.lap_kernel, padding=1)
            edge_stop = (0.25 + 0.75 * torch.exp(-0.55 * image_edge)).clamp(0.15, 1.0)
            logit_current = logit_current + self.curvature_strength * boundary * edge_stop * lap_logit
            curved = torch.sigmoid(logit_current).clamp(1e-4, 1.0 - 1e-4)

            current = anchor * decision0 + (1.0 - anchor) * curved
            current = current.clamp(1e-4, 1.0 - 1e-4)

        boundary = self._boundary_band(current)
        aa = F.conv2d(current, self.gauss_kernel, padding=1)
        current = current * (1.0 - 0.28 * boundary) + aa * (0.28 * boundary)
        current = torch.sigmoid(self.sharpen * _safe_logit(current))
        return current.clamp(0.0, 1.0), anchor
