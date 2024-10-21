# import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class PredictionModule:
    def __init__(self, modelPath='../models/lstm2.keras'):
        self.modelPath = modelPath
        self.model = keras.models.load_model(self.modelPath)
        self.scaler = MinMaxScaler()

    def predict(self, data: np.ndarray) -> dict:
        data_2d = data.reshape(-1, 3)
        self.scaler.fit(data_2d)
        scaled_data = self.scaler.transform(data_2d)
        scaled_data = scaled_data.reshape(1, 48, 3)
        prediction = self.model.predict(scaled_data)
        prediction = self.scaler.inverse_transform(prediction)
        return {
            'PM10': prediction[0, 0],
            'PM25': prediction[0, 1],
            'NO2': prediction[0, 2]
        }
