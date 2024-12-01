import tensorflow.keras as keras
import numpy as np
import concurrent.futures
import logging
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


class PredictionModule():
    def __init__(self, scalerFitInput: np.ndarray, modelPath='../models/lstm2.keras'):
        self.modelPath = modelPath
        self.model = keras.models.load_model(self.modelPath)
        self.scaler = StandardScaler().fit(scalerFitInput)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.logger = logging.getLogger(__name__)
        self.retrainingInProgress = False
        self.currentBestMetrics = self.initOriginalModelMetrics()

    def initOriginalModelMetrics(self):
        # MAE, MAPE, RMSE
        return [1.431499, 0.168695, 4.422874]

    def scoringFunction(self, metric: list) -> float:
        print(metric, type(metric))
        return metric[0] + 0.5 * metric[1] + 0.5 * metric[2]

    def predict(self, data: np.ndarray) -> dict:
        data_2d = data.reshape(-1, 3)
        scaled_data = self.scaler.transform(data_2d)
        scaled_data = scaled_data.reshape(1, 48, 3)
        prediction = self.model.predict(scaled_data)
        prediction = self.scaler.inverse_transform(prediction)
        return {
            'PM10': prediction[0, 0],
            'PM25': prediction[0, 1],
            'NO2': prediction[0, 2]
        }

    def singleRretrainRun(self, data: np.ndarray, labels: np.ndarray, epochs: int = 5):
        self.logger.info(f'Retraining model with {epochs} epochs')
        newModel = keras.models.load_model(self.modelPath)
        newModel.fit(data, labels, epochs=epochs)
        modelMAE, modelMAPE, modelRMSE = self.calculateMetrics(newModel.predict(data), labels)
        self.logger.info(f'MAE: {modelMAE}, MAPE: {modelMAPE}, RMSE: {modelRMSE}')
        return newModel, modelMAE, modelMAPE, modelRMSE

    def calculateMetrics(self, predictions, real_values):
        real_values = real_values.squeeze(axis=1)
        mae = mean_absolute_error(real_values[:, 0], predictions[:, 0])
        mape = mean_absolute_percentage_error(real_values[:, 1], predictions[:, 1])
        rmse = np.sqrt(mean_squared_error(real_values[:, 2], predictions[:, 2]))
        return [mae, mape, rmse]

    def retrainModel(self, data: np.ndarray, labels: np.ndarray):
        data_2d = data.reshape(-1, 3)
        scaled_data = self.scaler.transform(data_2d)
        scaled_data = scaled_data.reshape(data.shape)
        scaled_labels = self.scaler.transform(labels.reshape(-1, 3)).reshape(labels.shape)
        try1, mae1, mape1, rmse1 = self.singleRretrainRun(scaled_data, scaled_labels, epochs=5)
        try2, mae2, mape2, rmse2 = self.singleRretrainRun(scaled_data, scaled_labels, epochs=8)
        try3, mae3, mape3, rmse3 = self.singleRretrainRun(scaled_data, scaled_labels, epochs=10)

        models = [try1, try2, try3]
        metrics = [
            [mae1, mape1, rmse1],
            [mae2, mape2, rmse2],
            [mae3, mape3, rmse3]
        ]

        scores = [self.scoringFunction(m) for m in metrics]
        best_index = scores.index(min(scores))

        self.logger.info(f'Old metrics: {self.currentBestMetrics}; Batch best: {metrics[best_index]}')
        if scores[best_index] > self.scoringFunction(self.currentBestMetrics):
            return None, None

        return models[best_index], metrics[best_index]

    def retrain(self, data: np.ndarray, labels: np.ndarray):
        if self.retrainingInProgress:
            self.logger.info('Attempting to retrain, but retraining already in progress')
            return
        self.retrainingInProgress = True
        self.firstRetrain = True
        future = self.executor.submit(self.retrainModel, data, labels)
        future.add_done_callback(self.retrainCallback)

    def retrainCallback(self, future):
        try:
            bestModel, metrics = future.result()

            if bestModel is None:
                self.logger.info('Retraining successful: keeping old model')
                return

            self.model = bestModel
            self.currentBestMetrics = metrics
            bestModel.save('../models/lstm2_retrained.keras')
            self.modelPath = '../models/lstm2_retrained.keras'

            self.logger.info('Retraining successful: changing model')
        except Exception as e:
            self.logger.info(f"Retraining failed: {str(e)}")
        finally:
            self.retrainingInProgress = False
