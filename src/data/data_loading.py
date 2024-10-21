import numpy as np
import logging
import pandas as pd
import os
from .model_actions.NN_module import PredictionModule


class DataModule:
    def __init__(self):
        print(os.getcwd())
        self.dataPath = '../data/processed/EEA-SK-Ba-trend.csv'
        self.logger = logging.getLogger(__name__)
        self.idx = 48
        self.df = pd.read_csv(self.dataPath)
        self.pred_df = pd.DataFrame(
            np.nan,
            index=range(self.idx),
            columns=[
                'DatetimeBegin',
                'PM10 Concentration',
                'PM2.5 Concentration',
                'NO2 Concentration'
            ]
        )
        self.obsolete = True
        self.data = {
            'dates': self.df['DatetimeBegin'].tolist()[:self.idx + 1],
            'PM10': self.df['PM10 Concentration'].tolist()[:self.idx],
            'PM25': self.df['PM2.5 Concentration'].tolist()[:self.idx],
            'NO2': self.df['NO2 Concentration'].tolist()[:self.idx],
        }
        self.data['PM10pred'] = [None] * self.idx
        self.data['PM25pred'] = [None] * self.idx
        self.data['NO2pred'] = [None] * self.idx
        self.driftModule = DriftModule()
        self.predictionModeule = PredictionModule()
        self.nextCallCaount = 0

    def nextData(self) -> None:
        self.obsolete = False
        self.data['dates'] = self.df['DatetimeBegin'].tolist()[:self.idx + 1]
        self.data['PM10'] = self.df['PM10 Concentration'].tolist()[:self.idx]
        self.data['PM25'] = self.df['PM2.5 Concentration'].tolist()[:self.idx]
        self.data['NO2'] = self.df['NO2 Concentration'].tolist()[:self.idx]

        self.savePredictions(
            self.predictionModeule.predict(self.prepareDataForPrediction())
        )

        self.data['PM10pred'] = self.pred_df['PM10 Concentration'].tolist()
        self.data['PM25pred'] = self.pred_df['PM2.5 Concentration'].tolist()
        self.data['NO2pred'] = self.pred_df['NO2 Concentration'].tolist()
        self.logger.info('Data updated')

        self.nextCallCaount += 1
        if self.nextCallCaount % 16 == 0:
            self.driftModule.detectDrift(self.data)
            self.nextCallCaount = 0

    def savePredictions(self, data: dict) -> None:
        date = self.df['DatetimeBegin'].iloc[self.idx + 1]
        newRow = pd.DataFrame({
            'DatetimeBegin': [date],
            'PM10 Concentration': [data['PM10']],
            'PM2.5 Concentration': [data['PM25']],
            'NO2 Concentration': [data['NO2']]
        })
        self.pred_df = pd.concat(
            [self.pred_df, newRow],
            ignore_index=True
        )
        self.obsolete = True

    def prepareDataForPrediction(self) -> np.ndarray:
        return np.array([
            self.data['PM10'][-48:],
            self.data['PM25'][-48:],
            self.data['NO2'][-48:]
        ])

    def incrementIndex(self) -> None:
        self.idx += 1
        self.nextData()


class DriftModule:
    def __init__(self):
        pass

    def detectDrift(self, data: dict) -> dict:
        return {"message": False}
