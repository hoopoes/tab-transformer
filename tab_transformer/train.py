import os
from yacs.config import CfgNode

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from config import get_cfg_defaults

from core.bank import BankPreprocessor
from models.tab_transformer import TabTransformer
from utils.logging import make_logger
from utils.utils import get_device


logger = make_logger(name='tab-transformer trainer')

cfg = get_cfg_defaults()
cfg.merge_from_file(os.path.join(os.getcwd(), 'tab_transformer/configs/base_tab.yaml'))
cfg.freeze()

logger.info(f'configuration: \n {cfg}')


machine = BankPreprocessor(cfg, logger)
train_loader, val_loader, test_loader, map_records, num_class_per_category = machine.get_loader(
    train_ratio=0.7, val_ratio=0.2, batch_size=cfg.TRAIN.BATCH_SIZE)


class Learner(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.cfg = cfg

        device = get_device()

        self.model = TabTransformer(
            num_class_per_category=num_class_per_category,
            num_cont_features=cfg.DATA.NUM_CONT_FEATURES,
            hidden_size=cfg.MODEL.HIDDEN_SIZE,
            num_layers=cfg.MODEL.NUM_LAYERS,
            num_heads=cfg.MODEL.NUM_HEADS,
            attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
            ff_drop_rate=cfg.MODEL.FF_DROP_RATE,
            continuous_mean_std=torch.randn(cfg.DATA.NUM_CONT_FEATURES, 2)
        ).to(device)

    def forward(self, x):
        # forward: defines prediction/inference actions
        preds = self.model(x)
        return preds

    def training_step(self, batch, batch_idx):
        # training_step = training loop, independent of forward
        # batch: torch_geometric.data.batch.Batch
        preds = self.model(batch['x_cate'], batch['x_cont'])
        loss_func = torch.nn.BCELoss()
        loss = loss_func(preds, batch['labels'])

        self.log(
            name="train_loss", value=loss,
            prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch['x_cate'], batch['x_cont'])
        loss_func = torch.nn.BCELoss()
        loss = loss_func(preds, batch['labels'])

        self.log(
            name="val_loss", value=loss,
            prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        return optimizer


learner = Learner(cfg=cfg)
wandb_logger = WandbLogger(
    project='tab-transformer',
    name='test',
)


trainer = pl.Trainer(max_epochs=1, logger=wandb_logger, gpus=1)

trainer.fit(learner, train_loader, val_loader)

