from nn import *

net = ControlNN()

s = np.array([0,0])
mb_size = 30
num_mb = 1000
xs = np.random.uniform(-1,1, (mb_size*num_mb, 3))
ys = np.sum(xs,1)

net.graph_output(s)
for i in range(num_mb):
    if i%1000 == 0:
        print i
    sample_x, sample_y = xs[mb_size*i:mb_size*(i+1)], ys[mb_size*i:mb_size*(i+1)]
    net.train(sample_x, sample_y)

print net.get_best_a(s)
net.graph_output(s)
