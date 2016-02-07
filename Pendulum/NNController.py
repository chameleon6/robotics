import os
import sys
import time
import nn

matlab_state_file = os.getcwd() + '/matlab_state_file.out'
python_action_file = os.getcwd() + '/python_action_file.out'
count = 0

while True:
    print "iter", count
    start_time = time.time()
    while not os.path.isfile(matlab_state_file):
        if time.time() - start_time > 10:
            sys.exit()
        pass

    f = open(matlab_state_file, 'r')
    lines = f.readlines()
    f.close()
    if len(lines) < 2:
        #print "continuing"
        continue

    print lines
    temp = lines[0].rstrip()
    reward = float(temp)

    ss = lines[1].rstrip().split(' ')
    state = map(float, ss)
    os.remove(matlab_state_file)

    f = open(python_action_file, 'w')
    f.write('%s\n' % (count * 10))
    f.close()

    count += 1
