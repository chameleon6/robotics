import time
import logging
import numpy as np
import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def silence():
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr

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

    def toc(self, name, thresh=0.0):
        if name not in self.start_times:
            return

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
            # assert len(a) == 3
            assert a[1] == '='

            s = a[2]
            if "'" in s or '"' in s:
                assert s[0] == s[-1]
                assert s[0] in ['"', "'"]
                for i in s[1:-1]:
                    assert i != s[0]
                ans[a[0]] = s[1:-1]
            elif '.' in s:
                ans[a[0]] = float(s)
            else:
                ans[a[0]] = int(s)


    print 'conf read:', file_name
    print ans
    return ans

def mu_std(ts):
    mu_ts = np.mean(ts, 0)
    s_ts = np.std(ts, 0)
    return mu_ts, s_ts

def standardize(ts, ms_t):
    mu_ts, s_ts = ms_t
    return (ts - mu_ts)/s_ts

def unstandardize(ts, ms_t):
    mu_ts, s_ts = ms_t
    return ts * s_ts + mu_ts
