from nn import *

net = ControlNN()
s = np.array([-100,30])
print net.get_best_a(s)
net.graph_output(s)
