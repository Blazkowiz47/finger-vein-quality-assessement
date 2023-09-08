import torch
from torch.nn import Conv2d, Module, Parameter
from common.train_pipeline.backbone import IsotropicBackBone
from common.train_pipeline.predictor import Predictor

from common.train_pipeline.stem import Stem
from torch.nn.functional import adaptive_avg_pool2d
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


class IsoTropicVIG(Module):
    def __init__(self, opt_cfg) -> None:
        super(IsoTropicVIG, self).__init__()
        channels = opt_cfg.n_filters
        k = opt_cfg.k
        act = opt_cfg.act
        norm = opt_cfg.norm
        bias = opt_cfg.bias
        epsilon = opt_cfg.epsilon
        stochastic = opt_cfg.use_stochastic
        conv = opt_cfg.conv
        self.n_blocks = opt_cfg.n_blocks
        drop_path = opt_cfg.drop_path
        num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        print("dpr", dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)]  # number of knn's k
        print("num_knn", num_knn)
        max_dilation = 196 // max(num_knn)
        self.stem = Stem(
            img_shape=(1, 60, 120),
            in_dim=1,
            out_dim=channels,
            act=act,
        )
        self.backbone = IsotropicBackBone(
            n_blocks=self.n_blocks,
            channels=channels,
            act=act,
            norm=norm,
            conv=conv,
            stochastic=stochastic,
            num_knn=num_knn,
            use_dilation=opt_cfg.use_dilation,
            epsilon=epsilon,
            bias=bias,
            dpr=dpr,
            max_dilation=max_dilation,
        )
        self.predictor = Predictor(
            channels=channels,
            hidden_channels=opt_cfg.predictor_hidden_channels,
            act=act,
            dropout=opt_cfg.predictor_dropout,
        )
        self.pos_embed = Parameter(torch.zeros(1, channels, 4, 8))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs)
        x = x + self.pos_embed
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = adaptive_avg_pool2d(x, 1)
        x = self.predictor(x)
        return x


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 100,
        "input_size": (1, 60, 120),
        "pool_size": None,
        "crop_pct": 1,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "gnn_patch16_224": _cfg(
        crop_pct=0.9,
        input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


@register_model
def isotropic_vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptionalConfig:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn  # neighbor num (default:9)
            self.conv = "mr"  # graph conv layer {edge, mr}
            self.act = "gelu"  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = "batch"  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.n_blocks = 12  # number of basic blocks in the backbone
            self.n_filters = 256  # number of channels of deep features
            self.n_classes = num_classes  # Dimension of out_channels
            self.predictor_dropout = drop_rate  # dropout rate
            self.use_dilation = False  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.predictor_hidden_channels = 512

    opt_cfg = OptionalConfig(**kwargs)
    model = IsoTropicVIG(opt_cfg)
    model.default_cfg = default_cfgs["gnn_patch16_224"]
    return model
