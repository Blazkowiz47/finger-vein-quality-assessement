"""
Attention block.
"""
from dataclasses import dataclass
from typing import Optional
from torch.nn import Linear, Module, MultiheadAttention

# TODO:
# So input to this layer will be (B, C, H, W)
# C is our token embeddings length.
# Let N = H*W which are out total nodes or tokens.
# So convert input into (B,C,N)
# Perform Conv1d to get QKV
# Perform transpose on all and convert (B,N,C)
# Q,K,V should be of shape: (B, N, C)
# Perform self attention
# Perform transpose and convert to (B,C,N)
# Reshape to (B,C,H,W)


@dataclass
class AttentionBlockConfig:
    """
    Attention block config.
    """

    in_dim: int = 256
    num_heads: int = 4
    embed_dim: Optional[int] = None
    vdim: Optional[int] = None
    kdim: Optional[int] = None
    out_channels: Optional[int] = None
    dropout: float = 0.0
    bias: bool = True
    batch_first: bool = True


class AttentionBlock(Module):
    """
    An isotropic block with attention mechanism.
    """

    def __init__(self, config: AttentionBlockConfig) -> None:
        embed_dim = config.embed_dim or config.in_dim
        super(AttentionBlock, self).__init__()
        self.key = Linear(
            config.in_dim,
            config.kdim or config.in_dim,
            config.bias,
        )
        self.value = Linear(
            config.in_dim,
            config.vdim or config.in_dim,
            config.bias,
        )
        self.query = Linear(config.in_dim, embed_dim, config.bias)
        self.attention = MultiheadAttention(
            embed_dim,
            config.num_heads,
            config.dropout,
            config.bias,
            kdim=config.kdim,
            vdim=config.vdim,
            batch_first=config.batch_first,
        )

    def forward(self, inputs):
        """
        Forward pass.
        """
        batch, channels, height, width = inputs.shape
        shortcut = inputs
        inputs = inputs.reshape((batch, channels, -1))
        inputs = inputs.transpose(1, 2)
        key = self.key(inputs)
        value = self.value(inputs)
        query = self.query(inputs)
        output, _ = self.attention(query, key, value)
        output = output.transpose(1, 2)
        output = output.reshape((batch, channels, height, width))
        output = output + shortcut
        return output
