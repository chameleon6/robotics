import numpy as np

class TransitionContainer:
    def __init__(self, max_hist_len):
        self.container = []
        self.max_hist_len = max_hist_len

    def append(self, x):
        self.container.append(x)
        if len(self.container) > self.max_hist_len and len(self.container) % 100 == 0:
            self.container = self.container[100:]

    def random_sample(self, n):
        inds = np.random.choice(range(len(self.container)), n, True)
        return [self.container[i] for i in inds]
