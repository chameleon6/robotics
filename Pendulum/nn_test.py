from nn import *

net = ControlNN()

s = np.array([0.5,0.5])
mb_size = 10
num_mb = 1000
xs = np.random.uniform(-100,100, (mb_size*num_mb, 3))
ys = np.sum(xs,1)

#net.graph_output(s)
for i in range(num_mb):
    print i
    sample_x, sample_y = xs[mb_size*i:mb_size*(i+1)], ys[mb_size*i:mb_size*(i+1)]
    net.train(sample_x, sample_y)

#print net.get_best_a(s)
net.graph_output(s)
