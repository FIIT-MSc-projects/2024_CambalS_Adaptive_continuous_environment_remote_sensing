import numpy as np


class AnomalyModule:
    def __init__(self):
        pass

    def predict(self, data: dict) -> int | None:
        try:
            actual = [data["PM10"], data["PM25"], data["NO2"]]
            predicted = [
                data["PM10pred"],
                data["PM25pred"],
                data["NO2pred"],
            ]

            if len(actual) != len(predicted):
                raise ValueError("Actual and predicted data lengths do not match.")
            for real, pred in zip(actual, predicted):
                if len(real) != len(pred):
                    raise ValueError("Actual and predicted data lengths do not match.")
            errors = np.array(actual) - np.array(predicted)
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            z_scores = np.abs((errors - mean_error) / std_error)
            threshold = 3  # Z-score threshold for anomaly detection

            anomalies = z_scores > threshold
            return int(np.any(anomalies))
        except ValueError as ve:
            raise Exception(f"Value error: {str(ve)}")
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
