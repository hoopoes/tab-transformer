import os
from config import get_cfg_defaults

import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from models.tab_transformer import TabTransformer

from utils.logging import make_logger
from utils.utils import get_device


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
train_loader, val_loader, test_loader, map_records, num_class_per_category = machine.get_loader(
    train_ratio=0.7, val_ratio=0.2, batch_size=BATCH_SIZE)

model = TabTransformer(
    num_class_per_category=num_class_per_category,
    num_cont_features=cfg.DATA.NUM_CONT_FEATURES,
    hidden_size=cfg.MODEL.HIDDEN_SIZE,
    num_layers=cfg.MODEL.NUM_LAYERS,
    num_heads=cfg.MODEL.NUM_HEADS,
    attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
    ff_drop_rate=cfg.MODEL.FF_DROP_RATE,
    continuous_mean_std=torch.randn(cfg.DATA.NUM_CONT_FEATURES, 2)
)


# batch = next(iter(train_loader))
# preds = model(batch['x_cate'], batch['x_cont'])

