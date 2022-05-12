import numpy as np


class RLSEstimator:
    def __init__(self):
        self.x = None  # Initalize this with first measurement if None
        self.z = None
        self.h = None
        self.w = None
        self.r = None
        self.p = None

    
