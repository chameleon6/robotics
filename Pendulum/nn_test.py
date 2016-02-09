from nn import *
import time
import sys

start_time = time.time()
net = ControlNN()
print "compile time", time.time() - start_time

s = np.array([0.5, 0.5])
mb_size = 20
num_mb = 3000
n_test = 1000

def ys_from_xs(xs):
    #return np.sum(xs,1)[:,np.newaxis]
    ys = -(xs[:,0]**2 + xs[:,1] + (xs[:,2] - 0.5)**2)
    return ys[:, np.newaxis]

xs = np.random.uniform(-5,5, (mb_size*num_mb, 3))
ys = ys_from_xs(xs)

xs_test = np.random.uniform(-5,5, (n_test, 3))
ys_test = ys_from_xs(xs_test)

start_time = time.time()

for i in range(num_mb):
    sample_x, sample_y = xs[mb_size*i:mb_size*(i+1)], ys[mb_size*i:mb_size*(i+1)]
    if i%1000 == 0:
        print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)
    net.train(sample_x, sample_y)
    if i%1000 == 0:
        print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)

net.save_model()
print "train time", time.time() - start_time

print net.get_best_a(s)
net.graph_output(s)
