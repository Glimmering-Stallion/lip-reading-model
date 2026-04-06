"""PyTorch structural twin of the Rust VSRM (VsrModel in vsrm/vsrm.rs).

Used only for ONNX export (Burn has no native ONNX exporter). Weights are
PyTorch defaults unless you add a checkpoint loader later — not the trained
Burn checkpoint unless Phase C weight mapping exists.
"""


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock3d(nn.Module):
    """Mirrors LRM Rust ResidualBlock: conv1→GN→ReLU→conv2→GN, then ReLU(conv2_out + residual)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        norm_groups: int,
    ) -> None:
        super().__init__()
        kt, kh, kw = kernel_size
        pt, ph, pw = padding
        self.conv1 = nn.Conv3d(
            in_ch, out_ch, (kt, kh, kw), stride=stride, padding=(pt, ph, pw)
        )
        self.gn1 = nn.GroupNorm(norm_groups, out_ch)
        self.conv2 = nn.Conv3d(
            out_ch, out_ch, (kt, kh, kw), stride=1, padding=(pt, ph, pw)
        )
        self.gn2 = nn.GroupNorm(norm_groups, out_ch)
        nn.init.constant_(self.gn2.weight, 0.0)

        need_proj = in_ch != out_ch or any(s > 1 for s in stride)
        if need_proj:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x) if self.proj is not None else x
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return F.relu(x + residual)



class TcnBlock1d(nn.Module):
    """Mirrors TcnBlock: left-pad time, dilated Conv1d, LayerNorm over channels at each T."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, dilation=dilation, padding=0
        )
        self.ln1 = nn.LayerNorm(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, dilation=dilation, padding=0
        )
        self.ln2 = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout_p)
        self.proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x) if self.proj is not None else x
        p = self.padding
        x = F.pad(x, (p, 0))
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.ln1(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.pad(x, (p, 0))
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.ln2(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        return x + residual



class TemporalConvNetTwin(nn.Module):
    """Stack of TcnBlock1d with dilation 1, 2, 4, ... — matches TemporalConvNet::new."""

    def __init__(
        self,
        channels: tuple[int, int],
        kernel_size: int,
        num_layers: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        in_c, out_c = channels
        blocks: list[TcnBlock1d] = []
        cur = in_c
        for i in range(num_layers):
            dil = 1 << i
            blocks.append(
                TcnBlock1d(cur, out_c, kernel_size, dil, dropout_p)
            )
            cur = out_c
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x



class VsrTwin(nn.Module):
    """NCTHW in, NTV logits out — same pipeline as VsrModel::forward."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        frame_hw: tuple[int, int],
        norm_groups: int,
        vocab_size: int,
        tcn_kernel: int = 3,
        tcn_layers: int = 3,
        tcn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_hw = frame_hw
        k = (3, 3, 3)
        s = (1, 2, 2)
        pad = (1, 1, 1)

        oc = out_channels
        self.rb1 = ResidualBlock3d(in_channels, oc, k, s, pad, norm_groups)
        self.rb2 = ResidualBlock3d(oc, 2 * oc, k, s, pad, norm_groups)
        self.rb3 = ResidualBlock3d(2 * oc, 4 * oc, k, s, pad, norm_groups)

        rb3_out = 4 * oc
        # Rust uses AdaptiveAvgPool2d(4,4). TorchScript ONNX rejects that op when 4 does not divide H,W.
        # interpolate(..., mode="area") still lowers to adaptive_avg_pool2d in the graph.
        # Bilinear resize maps to ONNX Resize and exports reliably (slightly different math).
        self._pool_hw = (4, 4)
        flat = rb3_out * 4 * 4
        self.proj = nn.Linear(flat, hidden_dim)
        self.tcn1 = TemporalConvNetTwin(
            (hidden_dim, hidden_dim), tcn_kernel, tcn_layers, tcn_dropout
        )
        self.tcn2 = TemporalConvNetTwin(
            (hidden_dim, hidden_dim), tcn_kernel, tcn_layers, tcn_dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        n, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(n * t, c, h, w)
        x = F.interpolate(
            x, size=self._pool_hw, mode="bilinear", align_corners=False
        )
        x = x.view(n, t, -1)
        x = F.relu(self.proj(x))
        x = x.transpose(1, 2)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x.transpose(1, 2)
        return self.fc(x)
