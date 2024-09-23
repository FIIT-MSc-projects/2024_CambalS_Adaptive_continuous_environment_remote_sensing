import random
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import os
from drift_detection import DriftModule

class DataModule:
    def __init__(self):
        self.dataPath = 'static/EEA-SK-Ba-trend.csv'
        self.logger = logging.getLogger(__name__)
        self.df = pd.read_csv(self.dataPath)
        self.obsolete = True
        self.idx = 1
        self.data = {
            'dates': self.df['DatetimeBegin'].tolist()[:self.idx],
            'real': self.df['PM10 Concentration'].tolist()[:self.idx],
            'pred': self.df['PM2.5 Concentration'].tolist()[:self.idx],
        }
        self.driftModule = DriftModule()
        self.nextCallCaount = 0

    def nextData(self) -> None:
        self.obsolete = False
        self.idx += 1
        self.data = {
            'dates': self.df['DatetimeBegin'].tolist()[:self.idx],
            'real': self.df['PM10 Concentration'].tolist()[:self.idx],
            'pred': self.df['PM2.5 Concentration'].tolist()[:self.idx],
        }
        self.logger.info('Data updated')

        self.nextCallCaount += 1
        if self.nextCallCaount % 16 == 0:
            self.driftModule.detectDrift(self.data)
            self.nextCallCaount = 0