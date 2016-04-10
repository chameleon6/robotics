import numpy as np

m_ind_offset = 2

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def foot_coords(base_z, hip_angle, knee_angle):
    h = base_z - 0.5*(cos(hip_angle) + cos(hip_angle + knee_angle))
    rel_x = -0.5*(sin(hip_angle) + sin(hip_angle + knee_angle))
    return h, rel_x

def left_foot_coords(x):
    base_z = x[2-m_ind_offset]
    base_relative_pitch = x[3-m_ind_offset]
    left_knee_pin = x[5-m_ind_offset]

    h, rel_x = foot_coords(base_z, base_relative_pitch, left_knee_pin)
    return h, rel_x

def right_foot_coords(x):

    base_z = x[2-m_ind_offset]
    base_relative_pitch = x[3-m_ind_offset]
    hip_pin = x[7-m_ind_offset]
    right_knee_pin = x[8-m_ind_offset]

    h, rel_x = foot_coords(base_z, base_relative_pitch+hip_pin, right_knee_pin)
    return h, rel_x

def matlab_reward(x):
    left_h, left_x = left_foot_coords(x)
    right_h, right_x = right_foot_coords(x)
    if x[2-m_ind_offset] > 0.9 and x[2-m_ind_offset] < 1.05 and x[10-m_ind_offset] > 0:
        #num_x_steps = floor(x(1)/obj.reward_x_step);
        #if num_x_steps > last_reward_x_step & left_x * right_x < 0 & x(10) > 0.5
        #    last_reward_x_step = num_x_steps;
        #    t
        #    r = 1
        #    x
        #else
        #    r = 0;
        #end

        #r = -(c(1) - (left_x + right_x)/2)^2;

        if left_x * right_x < 0:
            return 1.0
        return 0.0

        r1 = -(x[10-m_ind_offset]-0.5)**2
        r2 = -(left_x+right_x)**2
        r = r1+r2
        return r
    else:
        return 0.0
