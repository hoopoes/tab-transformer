import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class GEGLU(nn.Module):
    """
    activation function which is a variant of GLU.
    reference: https://arxiv.org/pdf/2002.05202v1.pdf
    """
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def _trunc_normal_(x: torch.Tensor, mean: float = 0.0, std: float = 1.0):
    # from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class _Embedding(nn.Embedding):
    def __init__(self, ni: int, nf: int, std: float = 0.01):
        super(_Embedding, self).__init__(ni, nf)
        _trunc_normal_(self.weight.data, std=std)


class SharedEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            shared_embed: bool = True,
            shared_method: str = 'add',
            shared_embed_div: int = 8
    ):
        super().__init__()

        self.shared_method = shared_method

        assert shared_method in ['add', 'concat'], "only add or concat method is allowed"

        if shared_embed:
            if shared_method == 'add':
                shared_embed_dim = embedding_dim
                self.embed = _Embedding(num_embeddings, embedding_dim)
            else:
                shared_embed_dim = embedding_dim // shared_embed_div
                self.embed = _Embedding(num_embeddings, embedding_dim - shared_embed_dim)

            # one shared vector for each category
            self.shared_embed = nn.Parameter(torch.empty(1, 1, shared_embed_dim))
            _trunc_normal_(self.shared_embed.data, std=0.01)
        else:
            self.embed = _Embedding(num_embeddings, embedding_dim)
            self.shared_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b) -> (b, 1)
        out = self.embed(x).unsqueeze(1)
        if self.shared_embed is None:
            return out

        if self.shared_method == 'add':
            out += self.shared_embed
        else:
            # (b, num_category, embedding_dim)
            shared_embed = self.shared_embed.expand(out.shape[0], -1, -1)
            out = torch.cat((out, shared_embed), dim=-1)
        return out
