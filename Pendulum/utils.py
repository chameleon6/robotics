import time

start_times = {}

def tic(name):
    start_times[name] = time.time()

def toc(name, thresh=0.2):
    ans = time.time() - start_times[name]
    if ans > thresh:
        print "Global profile:", name, "took", ans, "seconds"

def profile_function(f, *args):
    start_time = time.time()
    output = f(*args)
    print "Function profile:", f.__name__, "took", time.time() - start_time, "seconds"
    return output

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


    print 'conf read:'
    print ans
    return ans
