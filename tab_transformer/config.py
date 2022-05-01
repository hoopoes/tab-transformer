from yacs.config import CfgNode as CN


_C = CN()

# directories
_C.ADDRESS = CN()
_C.ADDRESS.DATA = 'data/'
_C.ADDRESS.CHECK = 'checkpoints/'

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

_C.TRAIN.SCHEDULER = 'cos'

_C.TRAIN.FIRST_CYCLE_STEPS = 100
_C.TRAIN.CYCLE_MULT = 1.0
_C.TRAIN.MAX_LR = 0.1
_C.TRAIN.MIN_LR = 0.001
_C.TRAIN.WARMUP_STEPS = 0
_C.TRAIN.GAMMA = 1.0


def get_cfg_defaults():
    """
    get a yacs CfgNode object with default values
    """
    return _C.clone()
