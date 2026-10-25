from typing import Tuple
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _valid_num_groups(num_groups: int, channels: int) -> int:
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


def _safe_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.logit(x.clamp(eps, 1.0 - eps))


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class ConvGNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
                 padding: int = 1, groups: int = 1, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.gn = nn.GroupNorm(_valid_num_groups(num_groups, out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = ConvGNReLU(ch, ch, num_groups=num_groups)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_valid_num_groups(num_groups, ch), ch)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn2(self.conv2(self.conv1(x))) + x)
