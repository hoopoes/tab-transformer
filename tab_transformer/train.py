import os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from config import get_cfg_defaults

from core.bank_data import BankPreprocessor
from core.bank_learner import BankLearner

from utils.logging import make_logger


logger = make_logger(name='tab-transformer trainer')

cfg = get_cfg_defaults()
cfg.merge_from_file(os.path.join(os.getcwd(), 'tab_transformer/configs/base_tab.yaml'))
cfg.freeze()

logger.info(f'configuration: \n {cfg}')


machine = BankPreprocessor(cfg, logger)
train_loader, val_loader, test_loader, map_records, num_class_per_category = machine.get_loader(
    train_ratio=0.65, val_ratio=0.15, batch_size=cfg.TRAIN.BATCH_SIZE)


learner = BankLearner(cfg=cfg, num_class_per_category=num_class_per_category)
wandb_logger = WandbLogger(
    project='tab-transformer-experiment',
    name='bank: tt, v1',
)


trainer = pl.Trainer(max_epochs=2, logger=wandb_logger, gpus=1)

trainer.fit(learner, train_loader, val_loader)


# check
inputs = next(iter(test_loader))
preds = learner(inputs)
