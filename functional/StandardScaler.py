import numpy as np

class StandardScaler:
    def __init__(self):
        self.mu = 0
        self.std = 0

    def fit(self, x: np.ndarray):
        self.mu = np.mean(x)
        self.std = np.std(x)

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu)/self.std
