import torch
from torch import nn

from src.models.components.blocks import MLPBuilder


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
        activation: str = "relu",
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        :param activation: Activation function to use.
        :param batch_norm: Whether to use batch normalization.
        :param dropout: Dropout probability (0 to disable).
        """
        super().__init__()

        # Use MLPBuilder for consistent architecture
        hidden_sizes = [lin1_size, lin2_size, lin3_size]
        dropout_prob = dropout if dropout > 0 else None

        self.model = MLPBuilder.build(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout_prob,
            output_activation="none",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
