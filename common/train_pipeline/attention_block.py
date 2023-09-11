from typing import Optional
from torch.nn import Conv1d, Module, MultiheadAttention


class AttentionBlock(Module):
    """
    An isotropic block with attention mechanism.
    """

    def __init__(
        self,
        in_channels: int = 256,
        nodes: int = 32,
        num_heads: int = 2,
        kqv_filters=3,
        embed_dim: Optional[int] = None,
        vdim: Optional[int] = None,
        kdim: Optional[int] = None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        embed_dim = embed_dim or in_channels
        super(AttentionBlock, self).__init__()
        self.k = Conv1d(in_channels, kdim or in_channels, 3, 1, 1)
        self.v = Conv1d(in_channels, vdim or in_channels, 3, 1, 1)
        self.q = Conv1d(in_channels, embed_dim, 3, 1, 1)
        self.attention = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            bias,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape((B, C, -1))
        key = self.k(x)
        value = self.v(x)
        query = self.q(x)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = query.transpose(1, 2)
        o, w = self.attention(query, key, value)
        o = o.transpose(1, 2)
        o = o.reshape((B, C, H, W))
        return o


"""
So input to this layer will be (B, C, H, W)
C is our token embeddings length.
Let N = H*W which are out total nodes or tokens.

So convert input into (B,C,N)

Perform Conv1d to get QKV

Perform transpose on all and convert (B,N,C)

Q,K,V should be of shape: (B, N, C)

Perform self attention

Perform transpose and convert to (B,C,N)

Reshape to (B,C,H,W)
"""
