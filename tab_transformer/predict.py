import os

from config import get_cfg_defaults

import torch

from models.tab_transformer import TabTransformer

from core.bank_learner import BankLearner

from utils.logging import make_logger


logger = make_logger(name='tab-transformer trainer')

cfg = get_cfg_defaults()

# args
num_class_per_category = (2, 12, 3, 2, 12, 2, 4, 4, 3)
checkpoint = 'tt-epoch=4-val_auc=0.90.ckpt'

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

model.load_from_checkpoint(os.path.join(cfg.ADDRESS.CHECK), checkpoint)

learner = BankLearner(cfg=cfg, num_class_per_category=num_class_per_category)

learner.load_from_checkpoint(os.path.join(cfg.ADDRESS.CHECK, checkpoint))
