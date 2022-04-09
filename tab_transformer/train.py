import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config import get_cfg_defaults

from core.bank_data import BankPreprocessor
from core.bank_learner import BankLearner

from utils.logging import make_logger


# args
# parser = argparse.ArgumentParser(description='tab-transformer trainer parser')
# parser.add_argument('--config', '-c', type=str, default='v1', help='config file name')
# args = parser.parse_args()
# CONFIG_FILE = args.config

CONFIG_FILE = 'v3'

logger = make_logger(name='tab-transformer trainer')

cfg = get_cfg_defaults()
cfg.merge_from_file(os.path.join(os.getcwd(), f'tab_transformer/configs/{CONFIG_FILE}.yaml'))
cfg.freeze()

logger.info(f'configuration: \n {cfg}')

machine = BankPreprocessor(cfg, logger)
train_loader, val_loader, test_loader, map_records, num_class_per_category = machine.get_loader(
    train_ratio=0.65, val_ratio=0.15, batch_size=cfg.TRAIN.BATCH_SIZE)

learner = BankLearner(cfg=cfg, num_class_per_category=num_class_per_category)

wandb_logger = WandbLogger(
    project='tab-transformer-experiment',
    name=cfg.TRAIN.RUN_NAME,
)

# log gradients, parameters
wandb_logger.watch(learner.model, log='all', log_freq=100)

callbacks = [
    ModelCheckpoint(
        dirpath=cfg.ADDRESS.CHECK,
        filename='tt-{epoch}-{val_auc:.2f}',
        mode='max',
        every_n_val_epochs=1),
    EarlyStopping(
        monitor='val_auc',
        patience=cfg.TRAIN.PATIENCE,
        mode='max')
]

trainer = pl.Trainer(
    max_epochs=cfg.TRAIN.EPOCHS,
    gpus=1,
    logger=wandb_logger,
    callbacks=callbacks
)

trainer.fit(learner, train_loader, val_loader)

inputs = next(iter(test_loader))
preds = learner(inputs)
print(preds[0:5])
print(torch.where(preds >= 0.5)[0].shape, torch.where(preds < 0.5)[0].shape)

# for k, v in list(dict(learner.named_parameters()).items()):
#     print(k, v.detach().cpu().numpy().reshape(-1, 1)[0:5])

