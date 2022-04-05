from typing import Tuple, Dict
from yacs.config import CfgNode

import torch
from torchmetrics.functional import accuracy, precision, recall, auroc
import pytorch_lightning as pl

from models.tab_transformer import TabTransformer


class BankLearner(pl.LightningModule):
    def __init__(self, cfg: CfgNode, num_class_per_category: Tuple[int]):
        super().__init__()

        self.cfg = cfg

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

    def forward(self, inputs):
        preds = self.model(inputs['x_cate'], inputs['x_cont'])

        return preds

    def training_step(self, batch, batch_idx):
        loss, metrics = self._calculate_loss_and_metrics(batch, 'train')
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._calculate_loss_and_metrics(batch, 'val')
        self.log_dict(metrics)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-3)

        return optimizer

    def _calculate_loss_and_metrics(self, batch: Dict, prefix: str) -> Tuple[torch.Tensor, Dict]:
        preds = self.model(batch['x_cate'], batch['x_cont'])
        target = batch['labels']

        loss_func = torch.nn.BCELoss()
        loss = loss_func(preds, target)

        metrics = {
            f'{prefix}_loss': float(loss.detach().cpu().numpy()),
            f'{prefix}_acc': accuracy(preds=preds, target=target.type(torch.int32)),
            f'{prefix}_precision': precision(preds, target.type(torch.int32)),
            f'{prefix}_recall': recall(preds, target.type(torch.int32)),
            f'{prefix}_auc': auroc(preds, target.type(torch.int32))
        }

        return loss, metrics

