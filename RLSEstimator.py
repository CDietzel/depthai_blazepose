import numpy as np


class RLSEstimator:
    def __init__(self):
        self.x = []  # Initalize this with first measurement if None

    def estimate(self, measurement):
        if self.x != []:
            k = len(self.x)
            prev_x = self.x[-1]
            pred = prev_x + (1 / (k + 1)) * (measurement - prev_x)
            self.x.append(pred)
        else:
            self.x.append(measurement)
        return self.x[-1]

    def reset(self):
        self.x = []
