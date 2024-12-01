from river.drift import ADWIN


class DriftModule:
    def __init__(self, dataGatheringPeriod: int):
        self.driftDetectorPM10 = ADWIN(
            delta=0.00001,
            grace_period=50,
            clock=64 + dataGatheringPeriod
        )
        self.driftDetectorPM25 = ADWIN(
            delta=0.00001,
            grace_period=50,
            clock=64 + dataGatheringPeriod
        )
        self.driftDetectorNO2 = ADWIN(
            delta=0.00001,
            grace_period=50,
            clock=64 + dataGatheringPeriod
        )

    def detect(self, data: tuple) -> bool:
        self.driftDetectorPM10.update(data[0])
        self.driftDetectorPM25.update(data[1])
        self.driftDetectorNO2.update(data[2])

        if self.driftDetectorPM10.drift_detected or self.driftDetectorPM25.drift_detected or self.driftDetectorNO2.drift_detected:
            return True
        return False
