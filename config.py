from yacs.config import CfgNode as CN

_C = CN()

# directories
_C.ADDRESS = CN()
_C.ADDRESS.DOCS = 'docs/'

# model
_C.MODEL = CN()
_C.MODEL.NAME = 'base_tab'

# train
_C.TRAIN = CN()
_C.TRAIN.EMB_DIM = 128
