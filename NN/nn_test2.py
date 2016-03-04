from nn import *

net = ControlNN()
s = np.random.random((20,2))

a = np.array(map(net.get_best_a, s))[:,:,0,0]
a2 = np.array(net.get_best_a_p(s))[:,:,0].T

print a
print a2

