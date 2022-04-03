import os
from yacs.config import CfgNode

import pandas as pd

from core.base import BaseMachine


class AlbertMachine(BaseMachine):
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg

    def info(self):
        msg = """
        dataset url: https://automl.chalearn.org/data
        """
        return msg

    def load_data(self):
        data = pd.read_table(
            os.path.join(self.cfg.ADDRESS.DATA, 'albert/albert_train.data'),
            delimiter=' ', header=None)

        label = pd.read_table(
            os.path.join(self.cfg.ADDRESS.DATA, 'albert/albert_train.solution'),
            delimiter='\n', header=None)

        data[78] = label

        feature_types = pd.read_table(
            os.path.join(self.cfg.ADDRESS.DATA, 'albert/albert_feat.type'),
            delimiter='\n', header=None)

        return data, feature_types

    def preprocess(self):
        data, feature_types = self.load_data()


# temp
for i, ft in enumerate(feature_types):
    cast_type = 'float' if ft == 'Numerical' else 'int'
    data[i] = data[i].astype(cast_type)
