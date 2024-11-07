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
        newModel = keras.models.load_model(self.modelPath)
        newModel.fit(data, labels, epochs=epochs)
        modelMAE, modelMAPE, modelRMSE = self.calculateMetrics(newModel.predict(data), labels)
        return newModel, modelMAE, modelMAPE, modelRMSE

    def calculateMetrics(self, predictions, real_values):
        mae = mean_absolute_error(real_values[:, 0], predictions[:, 0])
        mape = mean_absolute_percentage_error(real_values[:, 1], predictions[:, 1])
        rmse = np.sqrt(mean_squared_error(real_values[:, 2], predictions[:, 2]))
        return [mae, mape, rmse]

    def retrainModel(self, data: np.ndarray, labels: np.ndarray):
        data_2d = data.reshape(-1, 3)
        scaled_data = self.scaler.transform(data_2d)
        scaled_data = scaled_data.reshape(1, 48, 3)
        scaled_labels = self.scaler.transform(labels)
        try1, mae1, mape1, rmse1 = self.singleRretrainRun(scaled_data, scaled_labels, epochs=5)
        try2, mae2, mape2, rmse2 = self.singleRretrainRun(scaled_data, scaled_labels, epochs=8)
        try3, mae3, mape3, rmse3 = self.singleRretrainRun(scaled_data, scaled_labels, epochs=10)
        # TODO: compare models, choose best, replace current one with the best

    def retrain(self, data: np.ndarray, labels: np.ndarray):
        if self.retrainingInProgress:
            self.logger.log('Attempting to retrain, but retraining already in progress')
            return
        self.retrainingInProgress = True
        future = self.executor.submit(self.retrainModel, data, labels)
        future.add_done_callback(self.retrainCallback)

    def retrainCallback(self, future):
        try:
            bestModel = future.result()

            if bestModel is None:
                self.logger.log('Retraining successful: keeping old model')
                return

            self.model = bestModel
            bestModel.save('../models/lstm2_retrained.keras')
            self.modelPath = '../models/lstm2_retrained.keras'

            self.logger.log('Retraining successful: changing model')
        except Exception as e:
            self.logger.log(f"Retraining failed: {str(e)}")
        finally:
            self.retrainingInProgress = False
