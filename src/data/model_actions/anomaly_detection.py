import numpy as np


class AnomalyModule:
    def __init__(self, z_thresh: float = 3.0):

        self.z_thresh = z_thresh

    def predict(self, data: dict) -> list[int | None]:
        features = ["PM10", "PM25", "NO2"]
        anomalies = []

        for idx, feat in enumerate(features):
            actual = np.array(data.get(feat, []))
            pred_key = f"{feat}pred"
            predicted = np.array(data.get(pred_key, []))

            if actual.size != predicted.size:
                raise ValueError(
                    f"Length mismatch for {feat}: actual={actual.size} vs predicted={predicted.size}"
                )

            if actual.size == 0:
                anomalies.append(None)
                continue

            errors = actual - predicted
            mean_err = np.mean(errors)
            std_err = np.std(errors)

            if std_err == 0:
                anomalies.append(None)
                continue

            z_scores = np.abs((errors - mean_err) / std_err)
            if z_scores[-1] > self.z_thresh:
                anomalies.append(idx + 1)
            else:
                anomalies.append(None)

        return anomalies
