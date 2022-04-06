"""
Notes
1) Missing values in categorical variables are not allowed.
 They must be filled with an additional category values
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

from .components import Residual, GEGLU, SharedEmbedding


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int = 32, multiplier: int = 4, drop_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size * multiplier * 2),
            GEGLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features=hidden_size * multiplier, out_features=hidden_size)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 32,
        num_heads: int = 8,
        dim_head: int = 16,
        drop_rate: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(
            in_features=hidden_size, out_features=inner_dim * 3, bias=False)
        self.to_out = nn.Linear(
            in_features=inner_dim, out_features=hidden_size)

        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dim_head: int,
        attn_drop_rate: float,
        ff_multiplier: int,
        ff_drop_rate: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Residual(
                    nn.Sequential(
                        nn.LayerNorm(normalized_shape=hidden_size),
                        Attention(hidden_size, num_heads, dim_head, attn_drop_rate)
                    )
                ),
                Residual(
                    nn.Sequential(
                        nn.LayerNorm(normalized_shape=hidden_size),
                        FeedForward(hidden_size, ff_multiplier, ff_drop_rate)
                    )
                ),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int = 170):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4*in_features, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4*in_features, out_features=2*in_features, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2*in_features, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class TabTransformer(nn.Module):
    def __init__(
        self,
        num_class_per_category: Tuple,
        num_cont_features: int,
        hidden_size: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_head: int = 16,
        continuous_mean_std: Optional[torch.Tensor] = None,
        attn_drop_rate: float = 0.0,
        ff_multiplier: int = 4,
        ff_drop_rate: float = 0.0
    ):
        """
        :param num_class_per_category: tuple containing the number of unique values within each category
        :param num_cont_features: number of continuous values
        :param hidden_size: hidden layer size
        :param num_layers: # layers of transformer
        :param num_heads: # heads of transformer
        :param dim_head: vector size of each head in attention layers
        :param continuous_mean_std: optional, normalize the continuous values before layer norm
        :param attn_drop_rate: drop out rate of attention net
        :param ff_multiplier: feature multiplier of feed-forward net
        :param ff_drop_rate: drop out rate of feed-forward net
        """
        super().__init__()
        assert all(map(lambda n: n > 0, num_class_per_category)), 'number of each category must be positive'

        # 1) categorical variables
        self.num_categories = len(num_class_per_category)
        self.num_category_classes = sum(num_class_per_category)

        # Shared Embedding
        self.embeds = nn.ModuleList()

        for num_class in num_class_per_category:
            self.embeds.append(
                SharedEmbedding(
                    num_embeddings=num_class, embedding_dim=hidden_size,
                    shared_embed=True, shared_method='add')
            )

        # 2) continuous variables
        if continuous_mean_std is not None:
            message = f'''
            continuous_mean_std must have a shape of ({num_cont_features}, 2)
            where the last dimension contains the mean and variance respectively
            '''
            assert continuous_mean_std.shape == (num_cont_features, 2), message

        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_cont_features)
        self.num_cont_features = num_cont_features

        self.transformer = Transformer(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_head=dim_head,
            attn_drop_rate=attn_drop_rate,
            ff_multiplier=ff_multiplier,
            ff_drop_rate=ff_drop_rate
        )

        # 3) final mlp
        num_features_mlp = (hidden_size * self.num_categories) + num_cont_features
        self.mlp = MLP(in_features=num_features_mlp)

    def forward(self, x_cate: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        # 1. categorical features
        x_cate = [embed(x_cate[:, i]) for i, embed in enumerate(self.embeds)]
        x_cate = torch.cat(x_cate, dim=1)

        x_cate = self.transformer(x_cate)
        x_cate = x_cate.flatten(1)

        # 2. continuous features
        if self.continuous_mean_std is not None:
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        x_cont = self.norm(x_cont)

        # 3. concatenation & mlp
        x = torch.cat((x_cate, x_cont), dim=-1)
        output = self.mlp(x)
        return output
