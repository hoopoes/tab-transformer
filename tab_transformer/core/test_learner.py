from typing import Tuple
from yacs.config import CfgNode

import torch
import pytorch_lightning as pl

from models.baseline_mlp import BaselineMLP
from utils.metrics import get_metric_collection


class BankLearner(pl.LightningModule):
    def __init__(self, cfg: CfgNode, num_class_per_category: Tuple[int]):
        super().__init__()

        self.cfg = cfg

        metrics = get_metric_collection()

        self.model = BaselineMLP(in_features=16)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, inputs):
        # forward: defines prediction/inference actions
        preds = self.model(torch.cat([inputs['x_cate'], inputs['x_cont']], dim=1))
        return preds

    def training_step(self, batch, batch_idx):
        # training_step = training loop, independent of forward
        # batch: torch_geometric.data.batch.Batch
        preds = self.model(torch.cat([batch['x_cate'], batch['x_cont']], dim=1))
        loss_func = torch.nn.BCELoss()
        loss = loss_func(preds, batch['labels'])

        train_metrics_log = self.train_metrics(preds, batch['labels'].type(torch.int32))
        self.log_dict(train_metrics_log)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(torch.cat([batch['x_cate'], batch['x_cont']], dim=1))
        loss_func = torch.nn.BCELoss()
        loss = loss_func(preds, batch['labels'])

        val_metrics_log = self.val_metrics(preds, batch['labels'].type(torch.int32))
        self.log_dict(val_metrics_log)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        return optimizer
