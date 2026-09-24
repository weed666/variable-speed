from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_receptive_field(kernel_size: int, dilations: list[int] | tuple[int, ...], convs_per_block: int = 2) -> int:
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")
    if convs_per_block < 1:
        raise ValueError("convs_per_block must be >= 1")
    if any(d < 1 for d in dilations):
        raise ValueError("all dilations must be >= 1")
    return 1 + convs_per_block * (kernel_size - 1) * sum(int(d) for d in dilations)


def get_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, hidden_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(hidden_channels, hidden_channels, kernel_size, dilation)
        self.activation1 = get_activation(activation)
        self.activation2 = get_activation(activation)
        self.output_activation = get_activation(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        y = self.conv1(x)
        y = self.activation1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.activation2(y)
        y = self.dropout2(y)
        return self.output_activation(residual + y)


@dataclass(frozen=True)
class TCNConfig:
    input_channels: int = 4
    hidden_channels: int = 32
    output_channels: int = 2
    kernel_size: int = 4
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    activation: str = "silu"


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 32,
        output_channels: int = 2,
        kernel_size: int = 4,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.config = TCNConfig(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dilations=tuple(int(d) for d in dilations),
            dropout=float(dropout),
            activation=activation,
        )
        blocks = []
        in_ch = input_channels
        for dilation in self.config.dilations:
            blocks.append(
                TCNResidualBlock(
                    in_channels=in_ch,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
            in_ch = hidden_channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_channels, output_channels)
        self.output_activation = nn.Tanh()
        self.reset_parameters()

    @property
    def receptive_field_samples(self) -> int:
        return compute_receptive_field(self.config.kernel_size, self.config.dilations, convs_per_block=2)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x=[B,C,T], got shape {tuple(x.shape)}")
        y = x
        for block in self.blocks:
            y = block(y)
        return y

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        features = self.extract_features(x)
        if return_sequence:
            temporal = features.transpose(1, 2)
            return self.output_activation(self.head(temporal)).transpose(1, 2)
        last = features[:, :, -1]
        return self.output_activation(self.head(last))

    def model_config_dict(self) -> dict:
        return {
            "input_channels": self.config.input_channels,
            "hidden_channels": self.config.hidden_channels,
            "output_channels": self.config.output_channels,
            "kernel_size": self.config.kernel_size,
            "dilations": list(self.config.dilations),
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "receptive_field_samples": self.receptive_field_samples,
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
