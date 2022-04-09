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

from models.tab_transformer import TabTransformer

from utils.logging import make_logger
from utils.metrics import get_metric_collection


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
train_loader, val_loader, test_data, map_records, num_class_per_category = machine.get_loader(
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

# for k, v in list(dict(learner.named_parameters()).items()):
#     print(k, v.detach().cpu().numpy().reshape(-1, 1)[0:5])

# test
"""
num_class_per_category = (3, 2, 4, 2, 12, 2, 3, 12, 4)
checkpoint = 'tt-epoch=9-val_auc=0.91.ckpt'

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

learner = BankLearner(cfg=cfg, num_class_per_category=num_class_per_category)
learner.load_from_checkpoint(os.path.join(cfg.ADDRESS.CHECK, checkpoint))
"""

metric_collection = get_metric_collection()

preds = torch.squeeze(learner.model(test_data['x_cate'], test_data['x_cont']), dim=1)
labels = test_data['labels']

metric_collection.update(preds.detach().cpu(), labels)
metrics = metric_collection.compute()
