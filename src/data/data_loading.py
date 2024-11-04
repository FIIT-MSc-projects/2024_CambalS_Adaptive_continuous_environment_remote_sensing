import numpy as np
import logging
import pandas as pd
import concurrent.futures
from .model_actions.NN_module import PredictionModule


class DataModule:
    def __init__(self):
        # logging setup
        self.logger = logging.getLogger(__name__)

        # data setup
        self.dataPath = '../data/processed/EEA-SK-Ba-trend.csv'
        self.idx = 48
        self.obsolete = True
        self.df = pd.read_csv(self.dataPath)
        self.data = {
            'dates': self.df['DatetimeBegin'].tolist()[:self.idx + 1],
            'PM10': self.df['PM10 Concentration'].tolist()[:self.idx],
            'PM25': self.df['PM2.5 Concentration'].tolist()[:self.idx],
            'NO2': self.df['NO2 Concentration'].tolist()[:self.idx],
        }
        self.data['PM10pred'] = [None] * self.idx
        self.data['PM25pred'] = [None] * self.idx
        self.data['NO2pred'] = [None] * self.idx

        # prediction setup
        self.driftModule = DriftModule()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.predictionModule = PredictionModule(
            self.getDataForScaler()
        )
        self.nextCallCount = 0

    def nextData(self) -> None:
        self.data['dates'] = self.df['DatetimeBegin'].tolist()[:self.idx + 1]
        self.data['PM10'] = self.df['PM10 Concentration'].tolist()[:self.idx]
        self.data['PM25'] = self.df['PM2.5 Concentration'].tolist()[:self.idx]
        self.data['NO2'] = self.df['NO2 Concentration'].tolist()[:self.idx]

        future = self.executor.submit(self.runPrediction)
        future.add_done_callback(self.savePredictionsCallback)

        self.logger.info('Data updated')
        self.obsolete = False

        # self.nextCallCaount += 1
        # if self.nextCallCaount % 16 == 0:
        #     self.driftModule.detectDrift(self.data)
        #     self.nextCallCount = 0

    def runPrediction(self):
        return self.predictionModule.predict(self.prepareDataForPrediction())

    def savePredictionsCallback(self, future):
        try:
            predictions = future.result()
            self.savePredictions(predictions)
            self.data['PM10pred'].append(self.df.loc[self.idx + 1, 'PM10 pred'])
            self.data['PM25pred'].append(self.df.loc[self.idx + 1, 'PM25 pred'])
            self.data['NO2pred'].append(self.df.loc[self.idx + 1, 'NO2 pred'])
            self.obsolete = False
            self.logger.info('Predictions updated')
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")

    def savePredictions(self, data: dict) -> None:
        self.df.loc[self.idx + 1, 'PM10 pred'] = float(data['PM10'])
        self.df.loc[self.idx + 1, 'PM25 pred'] = float(data['PM25'])
        self.df.loc[self.idx + 1, 'NO2 pred'] = float(data['NO2'])

    def prepareDataForPrediction(self) -> np.ndarray:
        return np.array([
            self.data['PM10'][-48:],
            self.data['PM25'][-48:],
            self.data['NO2'][-48:]
        ])

    def incrementIndex(self) -> None:
        self.idx += 1
        self.nextData()

    def getDataForScaler(self) -> np.ndarray:
        tmp = self.df[self.df['DatetimeBegin'] < '2020-01-01']
        return np.array([
            tmp['PM10 Concentration'].values,
            tmp['PM2.5 Concentration'].values,
            tmp['NO2 Concentration'].values
        ]).reshape(-1, 3)


class DriftModule:
    def __init__(self):
        pass

    def detectDrift(self, data: dict) -> dict:
        return {"message": False}
