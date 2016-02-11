import time

start_times = {}

def tic(name):
    start_times[name] = time.time()

def toc(name):
    ans = time.time() - start_times[name]
    print "Global profile:", name, "took", ans, "seconds"

def profile_function(f, *args):
    start_time = time.time()
    output = f(*args)
    print "Function profile:", f.__name__, "took", time.time() - start_time, "seconds"
    return output
