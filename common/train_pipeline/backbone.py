from enum import Enum
from typing import Any, List, Optional, Union
from torch.nn import Module, Sequential
from common.gcn_lib.torch_vertex import Grapher
from common.train_pipeline.ffn import FFN
from common.train_pipeline.attention_block import AttentionBlock


class SelfAttention(Enum):
    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"


class IsotropicBackBone(Module):
    def __init__(
        self,
        n_blocks: 5,
        channels=1,
        act: Any = "relu",
        norm=None,
        stochastic: bool = False,
        num_knn: Union[List[int], int] = 9,
        use_dilation=False,
        epsilon: float = 1e-4,
        bias: bool = True,
        dpr: Optional[List[float]] = None,
        max_dilation: float = 0,
        add_self_attention: Optional[SelfAttention] = None,
        conv="edge",
    ) -> None:
        super(IsotropicBackBone, self).__init__()
        self.n_blocks = n_blocks
        self.bias = bias
        self.epsilon: float = epsilon
        self.dpr: list[float] = dpr
        self.max_dilation = max_dilation
        self.act = act
        self.norm = norm
        self.stochastic = stochastic
        self.channels = channels
        self.conv = conv
        self.attention = add_self_attention
        if isinstance(num_knn, list):
            self.num_knn = num_knn
        else:
            self.num_knn = [num_knn for _ in range(self.n_blocks)]
        assert len(self.num_knn) == self.n_blocks, "Number of blocks and length of number of knn does not match."

        if use_dilation:
            self.backbone = self.build_backbone_with_dropout()
        else:
            self.backbone = self.build_backbone_without_dropout()

    def forward(self, x):
        # for i in range(self.n_blocks):
        x = self.backbone(x)
        return x

    def build_backbone_without_dropout(self) -> Sequential:
        layers: List[Sequential] = []
        for i in range(self.n_blocks):
            layers.append(
                Sequential(
                    AttentionBlock(256, 32, 2),
                    Grapher(
                        self.channels,
                        self.num_knn[i],
                        1,
                        self.conv,
                        self.act,
                        self.norm,
                        self.bias,
                        self.stochastic,
                        self.epsilon,
                        1,
                        drop_path=self.dpr[i],
                    ),
                    FFN(
                        self.channels,
                        self.channels * 4,
                        act=self.act,
                        drop_path=self.dpr[i],
                    ),
                )
            )
        return Sequential(*layers)

    def build_backbone_with_dropout(self) -> Sequential:
        layers: List[Sequential] = []
        for i in range(self.n_blocks):
            layers.append(
                Sequential(
                    AttentionBlock(256, 32, 2),
                    Grapher(
                        self.channels,
                        self.num_knn[i],
                        min(i // 4 + 1, self.max_dilation),
                        self.conv,
                        self.act,
                        self.norm,
                        self.bias,
                        self.stochastic,
                        self.epsilon,
                        1,
                        drop_path=self.dpr[i],
                    ),
                    FFN(
                        self.channels,
                        self.channels * 4,
                        act=self.act,
                        drop_path=self.dpr[i],
                    ),
                )
            )
        return Sequential(*layers)
