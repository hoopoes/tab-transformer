import torch
from models.tab_transformer import TabTransformer


model = TabTransformer(
    num_class_per_category=(10, 5, 6, 5, 8),
    num_cont_features=10,
    hidden_size=32,
    num_layers=6,
    num_heads=8,
    attn_drop_rate=0.1,
    ff_drop_rate=0.1,
    continuous_mean_std=torch.randn(10, 2)
)

x_cate = torch.randint(0, 5, (1, 5))
x_cont = torch.randn(1, 10)

pred = model(x_cate, x_cont)
