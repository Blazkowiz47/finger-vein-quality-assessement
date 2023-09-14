from enum import Enum
from typing import Any, Dict, List, Optional, Union
from torch.nn import Module, Sequential
from common.gcn_lib.torch_vertex import Grapher
from common.train_pipeline.backbone.ffn import FFN
from common.train_pipeline.backbone.attention_block import LinAttentionBlock


class SelfAttention(Enum):
    """
    Enum for self attention.
    """

    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"


class IsotropicBackBone(Module):
    """
    Isotropic Back bone.
    """

    def __init__(
        self,
        n_blocks: int = 5,
        block_type: str = "grapher",
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
        config: Optional[Dict[str, Any]] = None,
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
        self.config = config
        if isinstance(num_knn, list):
            self.num_knn = num_knn
        else:
            self.num_knn = [num_knn for _ in range(self.n_blocks)]
        assert (
            len(self.num_knn) == self.n_blocks
        ), "Number of blocks and length of number of knn doesn't match."

        if use_dilation:
            self.backbone = self.build_backbone_with_dropout()
        else:
            self.backbone = self.build_backbone_without_dropout()

    def forward(self, inputs):
        """
        Forward propogation.
        """
        # for i in range(self.n_blocks):
        inputs = self.backbone(inputs)
        return inputs

    def get_backbone_block(
        self,
        block_type: str,
        block_number: int,
    ) -> Sequential:
        if block_type == "grapher":
            return Sequential(
                Grapher(
                    self.channels,
                    self.num_knn[block_number],
                    1,
                    self.conv,
                    self.act,
                    self.norm,
                    self.bias,
                    self.stochastic,
                    self.epsilon,
                    drop_path=self.dpr[block_number],
                ),
                # LinAttentionBlock(self.channels, 32, 4),
                FFN(
                    self.channels,
                    self.channels * 4,
                    act=self.act,
                    drop_path=self.dpr[block_number],
                ),
            )

        if block_type == "self_attention":
            return Sequential(
                LinAttentionBlock(
                    self.channels,
                    self.config["num_nodes"],
                    self.config["num_heads"],
                ),
                FFN(
                    self.channels,
                    self.channels * 4,
                    act=self.act,
                    drop_path=self.dpr[block_number],
                ),
            )

        if block_type == "grapher_with_attention":
            return Sequential(
                Grapher(
                    self.channels,
                    self.num_knn[block_number],
                    1,
                    self.conv,
                    self.act,
                    self.norm,
                    self.bias,
                    self.stochastic,
                    self.epsilon,
                    drop_path=self.dpr[block_number],
                ),
                LinAttentionBlock(
                    self.channels,
                    self.config["num_nodes"],
                    self.config["num_heads"],
                ),
                FFN(
                    self.channels,
                    self.channels * 4,
                    act=self.act,
                    drop_path=self.dpr[block_number],
                ),
            )

    def build_backbone_without_dropout(self) -> Sequential:
        """
        Builds backbone with dropout.
        """
        layers: List[Sequential] = []
        for i in range(self.n_blocks):
            layers.append(
                Sequential(
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
                    # LinAttentionBlock(self.channels, 32, 4),
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
        """
        Builds backbone with dropout.
        """
        layers: List[Sequential] = []
        for i in range(self.n_blocks):
            layers.append(
                Sequential(
                    # Grapher(
                    #     self.channels,
                    #     self.num_knn[i],
                    #     min(i // 4 + 1, self.max_dilation),
                    #     self.conv,
                    #     self.act,
                    #     self.norm,
                    #     self.bias,
                    #     self.stochastic,
                    #     self.epsilon,
                    #     1,
                    #     drop_path=self.dpr[i],
                    # ),
                    LinAttentionBlock(self.channels, 32, 4),
                    FFN(
                        self.channels,
                        self.channels * 4,
                        act=self.act,
                        drop_path=self.dpr[i],
                    ),
                )
            )
        return Sequential(*layers)
