import random
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import os

class DataModule:
    def __init__(self):
        print(os.getcwd())
        self.dataPath = '../data/processed/EEA-SK-Ba-trend.csv'
        self.logger = logging.getLogger(__name__)
        self.df = pd.read_csv(self.dataPath)
        self.pred_df = pd.DataFrame(columns=['DatetimeBegin', 'PM10 Concentration', 'PM2.5 Concentration', 'NO2 Concentration'])
        self.obsolete = True
        self.idx = 1
        self.data = {
            'dates': self.df['DatetimeBegin'].tolist()[:self.idx + 1],
            'PM10': self.df['PM10 Concentration'].tolist()[:self.idx],
            'PM25': self.df['PM2.5 Concentration'].tolist()[:self.idx],
            'NO2': self.df['NO2 Concentration'].tolist()[:self.idx],
        }
        self.data['PM10pred'] = self.data['PM10'] + [random.uniform(-1, 1) for _ in range(1)]
        self.data['PM25pred'] = self.data['PM25'] + [random.uniform(-1, 1) for _ in range(1)]
        self.data['NO2pred'] = self.data['NO2'] + [random.uniform(-1, 1) for _ in range(1)]
        self.driftModule = DriftModule()
        self.nextCallCaount = 0

    def nextData(self) -> None:
        self.obsolete = False
        self.data['dates'] = self.df['DatetimeBegin'].tolist()[:self.idx + 1]
        self.data['PM10'] = self.df['PM10 Concentration'].tolist()[:self.idx]
        self.data['PM25'] = self.df['PM2.5 Concentration'].tolist()[:self.idx]
        self.data['NO2'] = self.df['NO2 Concentration'].tolist()[:self.idx]
        self.data['PM10pred'].append(self.data['PM10'][-1] - 4.0)
        self.data['PM25pred'].append(self.data['PM25'][-1] - 4.0)
        self.data['NO2pred'].append(self.data['NO2'][-1] - 4.0)
        self.logger.info('Data updated')

        self.nextCallCaount += 1
        if self.nextCallCaount % 16 == 0:
            self.driftModule.detectDrift(self.data)
            self.nextCallCaount = 0

    def getPredictions(self, idx: int, data: np.ndarray) -> None:
        date = datetime.strptime(data['dates'][idx], '%Y-%m-%d %H:%M:%S')
        date += timedelta(hours=1)
        self.pred_df = self.pred_df.append({
            'DatetimeBegin': date.strftime('%Y-%m-%d %H:%M:%S'),
            'PM10 Concentration': data[0, 0],
            'PM2.5 Concentration': data[0, 1],
            'NO2 Concentration': data[0, 2]
        }, ignore_index=True)
        self.obsolete = True

    def incrementIndex(self) -> None:
        self.idx += 1
        self.nextData()
        # self.getPredictions(self.idx, ...)

class DriftModule:
    def __init__(self):
        pass

    def detectDrift(self, data: dict) -> dict:
        return {"message": False}