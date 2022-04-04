import os
import wandb

from config import get_cfg_defaults

import numpy as np
import pandas as pd

import torch

from utils.logging import make_logger
from models.tab_transformer import TabTransformer


logger = make_logger(name='tab-transformer trainer')

cfg = get_cfg_defaults()
cfg.merge_from_file(os.path.join(os.getcwd(), 'tab_transformer/configs/base_tab.yaml'))
cfg.freeze()

logger.info(f'configuration: \n {cfg}')

# args
BATCH_SIZE = 32

# temp
from core.bank import BankPreprocessor

machine = BankPreprocessor(cfg, logger)
train_loader, val_loader, test_loader, map_records = machine.get_loader(
    train_ratio=0.7, val_ratio=0.2, batch_size=BATCH_SIZE)


# train
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

x_cate = torch.randint(0, 5, (BATCH_SIZE , 5))
x_cont = torch.randn(BATCH_SIZE , 10)

pred = model(x_cate, x_cont)

