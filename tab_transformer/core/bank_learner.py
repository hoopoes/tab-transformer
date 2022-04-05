from typing import Tuple
from yacs.config import CfgNode

import torch
import pytorch_lightning as pl

from models.tab_transformer import TabTransformer
from utils.metrics import get_metric_collection


class BankLearner(pl.LightningModule):
    def __init__(self, cfg: CfgNode, num_class_per_category: Tuple[int]):
        super().__init__()

        self.cfg = cfg

        metrics = get_metric_collection()

        self.model = TabTransformer(
            num_class_per_category=num_class_per_category,
            num_cont_features=cfg.DATA.NUM_CONT_FEATURES,
            hidden_size=cfg.MODEL.HIDDEN_SIZE,
            num_layers=cfg.MODEL.NUM_LAYERS,
            num_heads=cfg.MODEL.NUM_HEADS,
            attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
            ff_drop_rate=cfg.MODEL.FF_DROP_RATE,
            continuous_mean_std=torch.randn(cfg.DATA.NUM_CONT_FEATURES, 2)
        )

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, inputs):
        # forward: defines prediction/inference actions
        preds = self.model(inputs['x_cate'], inputs['x_cont'])
        return preds

    def training_step(self, batch, batch_idx):
        # training_step = training loop, independent of forward
        # batch: torch_geometric.data.batch.Batch
        preds = self.model(batch['x_cate'], batch['x_cont'])
        loss_func = torch.nn.BCELoss()
        loss = loss_func(preds, batch['labels'])

        train_metrics_log = self.train_metrics(preds, batch['labels'].type(torch.int32))
        self.log_dict(train_metrics_log)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch['x_cate'], batch['x_cont'])
        loss_func = torch.nn.BCELoss()
        loss = loss_func(preds, batch['labels'])

        val_metrics_log = self.val_metrics(preds, batch['labels'].type(torch.int32))
        self.log_dict(val_metrics_log)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        return optimizer
