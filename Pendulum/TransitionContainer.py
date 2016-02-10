import numpy as np

class TransitionContainer:
    def __init__(self):
        self.container = []

    def append(self, x):
        self.container.append(x)

    def random_sample(self, n):
        inds = np.random.randint(0, len(self.container), n)
        return [self.container[i] for i in inds]
