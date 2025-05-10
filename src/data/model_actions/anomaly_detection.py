import numpy as np


class AnomalyModule:
    def __init__(self, threshold: float = 1.2):
        self.threshold = threshold

    def predict(self, data: dict) -> list[int | None]:
        features = ["PM10", "PM25", "NO2"]
        anomalies = []

        for idx, feat in enumerate(features):
            actual = np.array(data.get(feat, []))
            pred_key = f"{feat}pred"
            predicted = data.get(pred_key, np.nan)

            if actual.size < 16:
                raise ValueError(
                    f"Less than 16 data points for feature {feat}. Anomaly detection requires at least 16 data points."
                )

            lastRealDatapoint = actual[-1]
            if predicted is not np.nan:
                if (
                    lastRealDatapoint * self.threshold
                    <= predicted
                    # or np.mean(actual) * (self.threshold + 0.15) <= predicted
                ):
                    anomalies.append(idx + 1)
                else:
                    anomalies.append(None)
            else:
                anomalies.append(None)

        return anomalies
