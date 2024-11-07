from river.drift import PageHinkley


class DriftModule:
    def __init__(self):
        self.driftDetectorPM10 = PageHinkley(
            min_instances=96,
            threshold=50.0
        )
        self.driftDetectorPM25 = PageHinkley(
            min_instances=96,
            threshold=50.0
        )
        self.driftDetectorNO2 = PageHinkley(
            min_instances=96,
            threshold=50.0
        )

    def detect(self, data: tuple) -> bool:
        pm10drift = self.driftDetectorPM10.update(data[0])
        pm25drift = self.driftDetectorPM25.update(data[1])
        no2drift = self.driftDetectorNO2.update(data[2])

        return pm10drift or pm25drift or no2drift
