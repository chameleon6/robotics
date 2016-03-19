import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

output = 'certainty_train_data.p'
xs, us = pickle.load(open('../NN/action_train_data.p', 'rb'))
print 'max/min u', np.max(us), np.min(us)
print 'max/min x', np.max(xs), np.min(xs)
print us.shape, xs.shape
n_good, n_s = xs.shape

max_x, min_x = np.max(xs, 0), np.min(xs, 0)
range_x = max_x - min_x
n_rand = len(xs)
rand_xs = np.ones((n_rand, 1)) * min_x[np.newaxis,:] + \
        np.ones((n_rand, 1)) * range_x[np.newaxis,:] * np.random.random((n_rand, n_s))

xs = np.concatenate((xs, rand_xs), 0)
us = np.concatenate((np.ones((n_good, 1)), np.zeros((n_rand, 1))), 0)

print 'max/min u', np.max(us), np.min(us)
print 'max/min x', np.max(xs), np.min(xs)
print us.shape, xs.shape
#pickle.dump((xs, us), open('../NN/simbicon_train_data.p', 'wb'))
pickle.dump((xs, us), open('../NN/' + output, 'wb'))
