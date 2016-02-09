from nn import *
import time

start_time = time.time()
net = ControlNN()
print "compile time", time.time() - start_time

s = np.array([0,0])
mb_size = 20
num_mb = 10000
n_test = 1000
xs = np.random.uniform(-1,1, (mb_size*num_mb, 3))
ys = np.sum(xs,1)

xs_test = np.random.uniform(-1,1, (n_test, 3))
ys_test = np.sum(xs_test,1)

#net.graph_output(s)

start_time = time.time()
for i in range(num_mb):
    if i%1000 == 0:
        print i, net.mse_q(xs_test, ys_test)
    sample_x, sample_y = xs[mb_size*i:mb_size*(i+1)], ys[mb_size*i:mb_size*(i+1)]
    net.train(sample_x, sample_y)

print "train time", time.time() - start_time

print net.get_best_a(s)
net.graph_output(s)
