"""Neural network building blocks and utilities."""

from typing import Optional

import torch
import torch.nn as nn


def linear_block(
    in_features: int,
    out_features: int,
    activation: str = "relu",
    batch_norm: bool = True,
    dropout: Optional[float] = None,
) -> nn.Sequential:
    """Create a linear block with optional batch norm, activation, and dropout.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation: Activation function name ("relu", "leaky_relu", "gelu", "none").
        batch_norm: Whether to include batch normalization.
        dropout: Dropout probability (None to disable).

    Returns:
        Sequential module containing the linear block.
    """
    layers = [nn.Linear(in_features, out_features)]

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))

    # Add activation
    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "leaky_relu":
        layers.append(nn.LeakyReLU())
    elif activation == "gelu":
        layers.append(nn.GELU())
    elif activation == "tanh":
        layers.append(nn.Tanh())
    elif activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif activation != "none":
        raise ValueError(f"Unknown activation: {activation}")

    # Add dropout
    if dropout is not None and dropout > 0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Optional[int] = None,
    activation: str = "relu",
    batch_norm: bool = True,
    dropout: Optional[float] = None,
) -> nn.Sequential:
    """Create a convolutional block with optional batch norm, activation, and dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding (auto-calculated if None).
        activation: Activation function name.
        batch_norm: Whether to include batch normalization.
        dropout: Dropout probability (None to disable).

    Returns:
        Sequential module containing the conv block.
    """
    if padding is None:
        padding = kernel_size // 2

    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    # Add activation
    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "leaky_relu":
        layers.append(nn.LeakyReLU())
    elif activation == "gelu":
        layers.append(nn.GELU())
    elif activation != "none":
        raise ValueError(f"Unknown activation: {activation}")

    # Add dropout
    if dropout is not None and dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers)


class MLPBuilder:
    """Builder for creating Multi-Layer Perceptron architectures."""

    @staticmethod
    def build(
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str = "relu",
        batch_norm: bool = True,
        dropout: Optional[float] = None,
        output_activation: str = "none",
    ) -> nn.Sequential:
        """Build an MLP with specified architecture.

        Args:
            input_size: Size of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Size of output layer.
            activation: Activation function for hidden layers.
            batch_norm: Whether to use batch normalization.
            dropout: Dropout probability for hidden layers.
            output_activation: Activation function for output layer.

        Returns:
            Sequential MLP model.
        """
        layers = []
        sizes = [input_size] + hidden_sizes

        # Hidden layers
        for i in range(len(sizes) - 1):
            layers.extend(
                linear_block(
                    sizes[i],
                    sizes[i + 1],
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
            )

        # Output layer (no batch norm, custom activation)
        layers.append(nn.Linear(sizes[-1], output_size))

        # Output activation
        if output_activation == "relu":
            layers.append(nn.ReLU())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif output_activation != "none":
            raise ValueError(f"Unknown output activation: {output_activation}")

        return nn.Sequential(*layers)


class CNNBuilder:
    """Builder for creating Convolutional Neural Network architectures."""

    @staticmethod
    def build_feature_extractor(
        input_channels: int,
        conv_configs: list[dict],
        global_pool: str = "adaptive_avg",
    ) -> nn.Sequential:
        """Build a CNN feature extractor.

        Args:
            input_channels: Number of input channels.
            conv_configs: List of conv layer configurations.
            global_pool: Global pooling type ("adaptive_avg", "adaptive_max", "none").

        Returns:
            Sequential CNN feature extractor.
        """
        layers = []
        in_channels = input_channels

        for config in conv_configs:
            out_channels = config["out_channels"]
            kernel_size = config.get("kernel_size", 3)
            stride = config.get("stride", 1)
            padding = config.get("padding", None)
            activation = config.get("activation", "relu")
            batch_norm = config.get("batch_norm", True)
            dropout = config.get("dropout", None)
            pool_size = config.get("pool_size", None)

            # Add conv block
            layers.extend(
                conv_block(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    activation,
                    batch_norm,
                    dropout,
                )
            )

            # Add pooling if specified
            if pool_size is not None:
                layers.append(nn.MaxPool2d(pool_size))

            in_channels = out_channels

        # Add global pooling
        if global_pool == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif global_pool == "adaptive_max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        elif global_pool != "none":
            raise ValueError(f"Unknown global pooling: {global_pool}")

        # Flatten
        layers.append(nn.Flatten())

        return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet-style architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """Initialize residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Convolution stride.
            downsample: Downsampling layer for skip connection.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttentionBlock(nn.Module):
    """Simple self-attention block."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        """Initialize attention block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Output tensor of same shape.
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
