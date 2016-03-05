import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

xs = np.zeros((0, 18))
us = np.zeros((0, 6))
for line in open('simbicon_output.out', 'r'):
    fname = line.strip()
    lines = open(fname, 'r').readlines()
    lines = map(lambda l: map(float, l.strip().split(' ')), lines)
    x = np.array(lines[::2])
    u = np.array(lines[1::2])
    xs = np.concatenate((xs, x), 0)
    us = np.concatenate((us, u), 0)

print 'max/min u', np.max(us), np.min(us)
print 'max/min x', np.max(xs), np.min(xs)
print us.shape, xs.shape
pickle.dump((xs, us), open('../NN/simbicon_train_data.p', 'wb'))
