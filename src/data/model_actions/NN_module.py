import tensorflow.keras as keras
import numpy as np
import concurrent.futures
import logging
import keras_tuner as kt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler


class PredictionModule:
    def __init__(self, scalerFitInput: np.ndarray, modelPath="../models/lstm2.keras"):
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
        # print(metric, type(metric))
        self.logger.info(
            f"Scoring metrics:\nMAE={metric[0]}, MAPE={metric[1]}, RMSE={metric[2]}"
        )
        return metric[0] + 0.5 * metric[1] + 0.5 * metric[2]

    def predict(self, data: np.ndarray) -> dict:
        data_2d = data.reshape(-1, 3)
        scaled_data = self.scaler.transform(data_2d)
        scaled_data = scaled_data.reshape(1, 48, 3)
        prediction = self.model.predict(scaled_data, verbose=0)
        prediction = self.scaler.inverse_transform(prediction)
        return {
            "PM10": prediction[0, 0],
            "PM25": prediction[0, 1],
            "NO2": prediction[0, 2],
        }

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
        scaled_labels = self.scaler.transform(labels.reshape(-1, 3)).reshape(
            labels.shape
        )
        self.logger.info("Retraining model")

        # Run the hyperparameter search
        best_hps = self.hyperParamSearch(scaled_data, scaled_labels)
        best_lr = best_hps.get("learning_rate")
        best_epochs = best_hps.Int("epochs", min_value=8, max_value=20, step=4)

        self.logger.info(f"Best Learning Rate: {best_lr}, Best Epochs: {best_epochs}")
        best_model = self.buildModel(best_hps)
        best_model.fit(scaled_data, scaled_labels, epochs=best_epochs)
        predictions = best_model.predict(scaled_data)

        best_model_mae, best_model_mape, best_model_rmse = self.calculateMetrics(
            predictions, scaled_labels
        )

        if self.scoringFunction(
            [best_model_mae, best_model_mape, best_model_rmse]
        ) > self.scoringFunction(self.currentBestMetrics):
            return None, None

        return best_model, [best_model_mae, best_model_mape, best_model_rmse]

    def retrain(self, data: np.ndarray, labels: np.ndarray):
        if self.retrainingInProgress:
            self.logger.info(
                "Attempting to retrain, but retraining already in progress"
            )
            return
        self.retrainingInProgress = True
        self.firstRetrain = True
        future = self.executor.submit(self.retrainModel, data, labels)
        future.add_done_callback(self.retrainCallback)

    def retrainCallback(self, future):
        try:
            bestModel, metrics = future.result()

            if bestModel is None:
                self.logger.info("Retraining successful: keeping old model")
                return

            self.model = bestModel
            self.currentBestMetrics = metrics
            bestModel.save("../models/lstm2_retrained.keras")
            self.modelPath = "../models/lstm2_retrained.keras"

            self.logger.info("Retraining successful: changing model")
        except Exception as e:
            self.logger.info(f"Retraining failed: {str(e)}")
        finally:
            self.retrainingInProgress = False

    def buildModel(self, hp: kt.HyperParameters):
        model = keras.models.load_model(self.modelPath)

        optimizer = keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[5e-4, 1e-3, 5e-3, 1e-2])
        )
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mape"])
        return model

    def hyperParamSearch(
        self, data: np.ndarray, labels: np.ndarray
    ) -> kt.HyperParameters:
        tuner = kt.RandomSearch(
            self.buildModel,
            max_trials=10,
            executions_per_trial=1,
            objective=kt.Objective("loss", direction="min"),
            # max_epochs=15,
            directory="random_search",
            project_name="lstm_lr_epochs_tuning",
        )
        tuner.search(data, labels, validation_split=0.2, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps
