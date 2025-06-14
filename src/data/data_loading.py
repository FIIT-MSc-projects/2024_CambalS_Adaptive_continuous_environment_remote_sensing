import numpy as np
import logging
import pandas as pd
import concurrent.futures
from .model_actions.NN_module import PredictionModule
from .model_actions.drift_detection import DriftModule
from .model_actions.anomaly_detection import AnomalyModule


class DataModule:
    def __init__(self):
        # logging setup
        self.logger = logging.getLogger(__name__)

        # data setup
        self.dataPath = "../data/processed/EEA-SK-Ba-trend.csv"
        # self.idx = 48
        self.idx = 128
        self.dataGatheringPeriod = 64
        self.obsolete = True
        self.df = pd.read_csv(self.dataPath)
        self.data = {
            "dates": self.df["DatetimeBegin"].tolist()[: self.idx + 2],
            "PM10": self.df["PM10 Concentration"].tolist()[: self.idx],
            "PM25": self.df["PM2.5 Concentration"].tolist()[: self.idx],
            "NO2": self.df["NO2 Concentration"].tolist()[: self.idx],
        }
        self.data["PM10pred"] = [None] * (self.idx + 1)
        self.data["PM25pred"] = [None] * (self.idx + 1)
        self.data["NO2pred"] = [None] * (self.idx + 1)
        self.data["retraining"] = [None] * (self.idx + 1)
        self.data["anomaly"] = {
            "PM10": [None] * (self.idx + 1),
            "PM25": [None] * (self.idx + 1),
            "NO2": [None] * (self.idx + 1),
        }

        # prediction setup
        self.driftModule = DriftModule(self.dataGatheringPeriod)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.predictionModule = PredictionModule(self.getDataForScaler())
        self.anomalyModule = AnomalyModule()
        self.nextCallCount = 0

    def nextData(self) -> None:
        if self.idx >= 128 + 17:
            future = self.executor.submit(self.runAnomalyDetection)
            future.add_done_callback(self.runAnomalyDetectionCallback)
        else:
            self.data["anomaly"]["PM10"].append(None)
            self.data["anomaly"]["PM25"].append(None)
            self.data["anomaly"]["NO2"].append(None)

        future = self.executor.submit(self.runPrediction)
        future.add_done_callback(self.savePredictionsCallback)

        self.obsolete = False

        isDriftPresent = self.driftModule.detect(
            (self.data["PM10"][-1], self.data["PM25"][-1], self.data["NO2"][-1])
        )
        if isDriftPresent:
            self.logger.info("Drift detected, attempting to retrain model")
            self.data["retraining"].append(1)
            features, labels = self.prepareDataFotRetraining()
            self.predictionModule.retrain(features, labels)
        else:
            self.data["retraining"].append(None)

    def runAnomalyDetection(self) -> None:
        return self.anomalyModule.predict(
            {
                "PM10": self.data["PM10"][-16:],
                "PM25": self.data["PM25"][-16:],
                "NO2": self.data["NO2"][-16:],
                "PM10pred": self.data["PM10pred"][-2],
                "PM25pred": self.data["PM25pred"][-2],
                "NO2pred": self.data["NO2pred"][-2],
            }
        )

    def runAnomalyDetectionCallback(self, future) -> None:
        try:
            anomaly = future.result()
            self.data["anomaly"]["PM10"].append(anomaly[0])
            self.data["anomaly"]["PM25"].append(anomaly[1])
            self.data["anomaly"]["NO2"].append(anomaly[2])
            if (
                anomaly[0] is not None
                or anomaly[1] is not None
                or anomaly[2] is not None
            ):
                pollutant = (
                    ["PM10", "PM25", "NO2"][anomaly[0] - 1]
                    if anomaly[0] is not None
                    else (
                        ["PM10", "PM25", "NO2"][anomaly[1] - 1]
                        if anomaly[1] is not None
                        else ["PM10", "PM25", "NO2"][anomaly[2] - 1]
                    )
                )
                self.logger.info(
                    f"Anomaly detected for pollutant: {pollutant} on index {self.idx}"
                )
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")

    def runPrediction(self):
        return self.predictionModule.predict(self.prepareDataForPrediction())

    def savePredictionsCallback(self, future):
        try:
            predictions = future.result()
            self.savePredictions(predictions)
            self.data["PM10pred"].append(self.df.loc[self.idx + 1, "PM10 pred"])
            self.data["PM25pred"].append(self.df.loc[self.idx + 1, "PM25 pred"])
            self.data["NO2pred"].append(self.df.loc[self.idx + 1, "NO2 pred"])
            self.obsolete = False
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
        finally:
            self.data["dates"] = self.df["DatetimeBegin"].tolist()[: self.idx + 2]
            self.data["PM10"] = self.df["PM10 Concentration"].tolist()[: self.idx]
            self.data["PM25"] = self.df["PM2.5 Concentration"].tolist()[: self.idx]
            self.data["NO2"] = self.df["NO2 Concentration"].tolist()[: self.idx]

    def savePredictions(self, data: dict) -> None:
        self.df.loc[self.idx + 1, "PM10 pred"] = float(data["PM10"])
        self.df.loc[self.idx + 1, "PM25 pred"] = float(data["PM25"])
        self.df.loc[self.idx + 1, "NO2 pred"] = float(data["NO2"])

    def prepareDataForPrediction(self) -> np.ndarray:
        return np.array(
            [self.data["PM10"][-48:], self.data["PM25"][-48:], self.data["NO2"][-48:]]
        )

    def incrementIndex(self) -> None:
        self.idx += 1
        if (self.idx - 128) % 20 == 0:
            self.logger.info(f"Idx: {self.idx - 128}")
        self.nextData()

    def getDataForScaler(self) -> np.ndarray:
        tmp = self.df[self.df["DatetimeBegin"] < "2020-01-01"]
        return np.array(
            [
                tmp["PM10 Concentration"].values,
                tmp["PM2.5 Concentration"].values,
                tmp["NO2 Concentration"].values,
            ]
        ).reshape(-1, 3)

    def prepareDataFotRetraining(
        self, n_past: int = 48, n_future: int = 1
    ) -> np.ndarray:
        data = np.array(
            [
                self.data["PM10"][-self.dataGatheringPeriod :],
                self.data["PM25"][-self.dataGatheringPeriod :],
                self.data["NO2"][-self.dataGatheringPeriod :],
            ]
        ).T

        features = []
        labels = []
        for i in range(len(data) - n_past - n_future + 1):
            features.append(data[i : i + n_past])
            labels.append(data[i + n_past : i + n_past + n_future])
        features = np.array(features)
        labels = np.array(labels)

        return features, labels

    def test(self):
        features, labels = self.prepareDataFotRetraining()
        self.predictionModule.retrain(features, labels)
