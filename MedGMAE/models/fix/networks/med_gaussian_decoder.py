import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block


class MedGaussianViTDecoder(nn.Module):
    def __init__(
        self,
        num_gaussians=512,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        patch_size=None,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.position_head = nn.Linear(embed_dim, 3)

        self.scale_head = nn.Linear(embed_dim, 3)

        self.rotation_head = nn.Linear(embed_dim, 4)

        self.density_head = nn.Linear(embed_dim, 1)

        self._custom_init_heads()

        self.apply(self._init_weights)

    def _custom_init_heads(self):

        nn.init.xavier_uniform_(self.position_head.weight)
        nn.init.zeros_(self.position_head.bias)

        nn.init.normal_(self.scale_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.scale_head.bias, -1.386)

        nn.init.xavier_uniform_(self.rotation_head.weight)
        nn.init.zeros_(self.rotation_head.bias)

        nn.init.xavier_uniform_(self.density_head.weight)
        nn.init.constant_(self.density_head.bias, -0.405)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m not in [
                self.position_head,
                self.scale_head,
                self.rotation_head,
                self.density_head,
            ]:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):

        x = self.forward_features(x)

        gaussian_tokens = x[:, 1 : self.num_gaussians + 1]

        positions = torch.sigmoid(self.position_head(gaussian_tokens))

        scales = torch.sigmoid(self.scale_head(gaussian_tokens))

        rotations = self.rotation_head(gaussian_tokens)

        rotations = nn.functional.normalize(rotations, dim=-1)

        densities = torch.sigmoid(self.density_head(gaussian_tokens))

        return positions, scales, rotations, densities


def med_gaussian_decoder_base(**kwargs):
    model = MedGaussianViTDecoder(embed_dim=384, depth=8, num_heads=12, **kwargs)
    return model


def med_gaussian_decoder_large(**kwargs):
    model = MedGaussianViTDecoder(embed_dim=528, depth=8, num_heads=16, **kwargs)
    return model
