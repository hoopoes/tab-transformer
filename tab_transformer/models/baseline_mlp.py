import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, in_features: int = 170):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4 * in_features, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4 * in_features, out_features=2 * in_features, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2 * in_features, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)
