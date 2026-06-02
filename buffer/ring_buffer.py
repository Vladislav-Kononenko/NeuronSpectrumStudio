import numpy as np

class RingBuffer:
    def __init__(self, n_channels: int, max_samples: int):
        self.n_channels = n_channels
        self.max_samples = max_samples
        self.data = np.zeros((n_channels, max_samples), dtype=float)
        self.timestamps = np.zeros(max_samples, dtype=float)
        self.index = 0
        self.filled = False

    def append(self, sample, timestamp):
        self.data[:, self.index] = sample
        self.timestamps[self.index] = timestamp
        self.index = (self.index + 1) % self.max_samples
        if self.index == 0:
            self.filled = True

    def is_ready(self):
        return self.filled

    def get_all(self):
        if not self.filled:
            return self.data[:, :self.index], self.timestamps[:self.index]
        idx = self.index
        data = np.concatenate((self.data[:, idx:], self.data[:, :idx]), axis=1)
        ts = np.concatenate((self.timestamps[idx:], self.timestamps[:idx]))
        return data, ts