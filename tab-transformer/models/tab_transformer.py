from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

from utils.utils import exists, default


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dim_head=16,
        dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            num_tokens: int,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            dim_head: int,
            attn_drop_rate: float,
            ff_drop_rate: float
    ):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, hidden_size)
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(
                    hidden_size,
                    Attention(hidden_size, num_heads=num_heads, dim_head=dim_head, dropout=attn_drop_rate))),
                Residual(PreNorm(
                    hidden_size,
                    FeedForward(hidden_size, dropout=ff_drop_rate))),
            ]))

    def forward(self, x):
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []

        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_class_per_category: Tuple,
        num_cont_features,
        hidden_size,
        num_layers,
        num_heads,
        dim_head: int = 16,
        dim_out: int = 1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens: int = 2,
        continuous_mean_std=None,
        attn_drop_rate: float = 0.0,
        ff_drop_rate: float = 0.0
    ):
        """
        :param num_class_per_category: tuple containing the number of unique values within each category
        :param num_cont_features: number of continuous values
        :param hidden_size:
        :param num_layers:
        :param num_heads:
        :param dim_head:
        :param dim_out:
        :param mlp_hidden_mults:
        :param mlp_act:
        :param num_special_tokens:
        :param continuous_mean_std:
        :param attn_drop_rate:
        :param ff_drop_rate:
        """
        super().__init__()
        assert all(map(lambda n: n > 0, num_class_per_category)), 'number of each category must be positive'

        # categorical variables
        self.num_categories = len(num_class_per_category)
        self.num_unique_categories = sum(num_class_per_category)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids
        # to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(num_class_per_category)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous variables
        if exists(continuous_mean_std):
            message = '''
            continuous_mean_std must have a shape of ({num_cont_features}, 2)
            where the last dimension contains the mean and variance respectively
            '''
            assert continuous_mean_std.shape == (num_cont_features, 2), message

        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_cont_features)
        self.num_cont_features = num_cont_features

        self.transformer = Transformer(
            num_tokens=total_tokens,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_head=dim_head,
            attn_drop_rate=attn_drop_rate,
            ff_drop_rate=ff_drop_rate
        )

        # mlp to logits
        input_size = (hidden_size * self.num_categories) + num_cont_features
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_cate, x_cont):
        # assert x_cate.shape[-1] == self.num_categories,
        # f'you must pass in {self.num_categories} values for your categories input'
        x_cate += self.categories_offset

        x = self.transformer(x_cate)

        flat_cate = x.flatten(1)

        # assert x_cont.shape[1] == self.num_cont_features,
        # f'you must pass in {self.num_cont_features} values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        x = torch.cat((flat_cate, normed_cont), dim=-1)
        output = self.mlp(x)
        return output


cont_mean_std = torch.randn(10, 2)


model = TabTransformer(
    num_class_per_category=(10, 5, 6, 5, 8),
    num_cont_features=10,
    hidden_size=32,
    dim_out=1,
    num_layers=6,
    num_heads=8,
    attn_drop_rate=0.1,
    ff_drop_rate=0.1,
    mlp_hidden_mults=(4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act=nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std=cont_mean_std # (optional) - normalize the continuous values before layer norm
)

x_cate = torch.randint(0, 5, (1, 5))
x_cont = torch.randn(1, 10)

pred = model(x_cate, x_cont)
