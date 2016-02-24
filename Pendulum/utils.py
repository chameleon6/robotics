import time
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

class Profiler:

    def __init__(self):
        self.start_times = {}
        self.runtime_stats = {}
        self.errs = []

    def tic(self, name):
        self.start_times[name] = time.time()

    def toc(self, name, thresh=0.2):
        ans = time.time() - self.start_times[name]
        if name not in self.runtime_stats:
            self.runtime_stats[name] = []
        self.runtime_stats[name].append(ans)

        if ans > thresh:
            print "Global profile:", name, "took", ans, "seconds"

    def log_err(self, err_msg):
        self.errs.append(err_msg)

    def profile_function(self, f, *args):
        start_time = time.time()
        output = f(*args)
        print "Function profile:", f.__name__, "took", time.time() - start_time, "seconds"
        return output

    def print_time_stats(self):
        print "Aggregate runtime stats:"
        for name,l in self.runtime_stats.iteritems():
            print name, "took avg time", sum(l)/len(l), "for", len(l), "runs"

def read_conf(file_name):
    ans = {}
    with open(file_name) as f:
        for line in f.readlines():
            a = line.strip().split(' ')
            if a[0] == '' or a[0] == '#':
                continue
            assert len(a) == 3
            assert a[1] == '='
            if '.' in a[2]:
                ans[a[0]] = float(a[2])
            else:
                ans[a[0]] = int(a[2])


    print 'conf read:', file_name
    print ans
    return ans
