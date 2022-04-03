import os
from logging import Logger
from yacs.config import CfgNode

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, MinMaxScaler

from core.base import BaseMachine


class BankMachine(BaseMachine):
    def __init__(
            self,
            cfg: CfgNode,
            logger: Logger
    ):
        self.cfg = cfg
        self.logger = logger

    def info(self):
        msg = "dataset url: https://archive.ics.uci.edu/ml/datasets/bank+marketing"
        return msg

    def load_data(self):
        self.logger.info('load_data')

        data = pd.read_csv(
            os.path.join(self.cfg.ADDRESS.DATA, 'bank/bank-full.csv'),
            sep=';', header=0)

        return data

    def preprocess(self, verbose: bool = True):
        self.logger.info('preprocess')

        data = self.load_data()

        cont_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        cate_features = list(set(data.columns).difference(set(cont_features)))
        cate_features.remove('y')

        map_records = {}

        # 1. categorical features mapping
        for f in cate_features:
            if f == 'month':
                month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec']
                map_dict = {m: i for i, m in enumerate(month_list)}
                data[f] = data[f].map(map_dict)
            else:
                all_class = sorted(data[f].unique().tolist())
                map_dict = {prev: new for new, prev in enumerate(all_class)}
                data[f] = data[f].map(map_dict)

            map_records[f] = map_dict

        self.logger.info("mapped categorical features to integer index")

        # 2. scale continuous features
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data[cont_features])

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(scaled)

        data = data.drop(cont_features, axis=1)
        data = pd.concat([data, pd.DataFrame(scaled, columns=cont_features)], axis=1)

        self.logger.info("scaled continuous features")

        # 3. label mapping
        data['y'] = data['y'].map({'yes': 1, 'no': 0})

        data = data[cate_features + cont_features + ['y']]

        additional_info = [map_records, cate_features, cont_features]

        return data, additional_info

