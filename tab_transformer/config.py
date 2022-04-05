from yacs.config import CfgNode as CN


_C = CN()

# directories
_C.ADDRESS = CN()
_C.ADDRESS.DATA = 'tab_transformer/data/'
_C.ADDRESS.CHECK = 'tab_transformer/checkpoints/'

# data
_C.DATA = CN()
_C.DATA.NUM_CONT_FEATURES = 7

# model
_C.MODEL = CN()
_C.MODEL.NAME = 'base_tab'
_C.MODEL.HIDDEN_SIZE = 32
_C.MODEL.NUM_LAYERS = 6
_C.MODEL.NUM_HEADS = 8
_C.MODEL.ATTN_DROP_RATE = 0.1
_C.MODEL.FF_DROP_RATE = 0.1

# train
_C.TRAIN = CN()
_C.TRAIN.RUN_NAME = 'v1'
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.EPOCHS = 5
_C.TRAIN.PATIENCE = 2


def get_cfg_defaults():
    """
    get a yacs CfgNode object with default values
    """
    return _C.clone()
