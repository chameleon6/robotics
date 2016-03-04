f = open('NNController.py', 'r')
s = f.read()

to_replace = [
'profiler',
'matlab_state_file',
'python_action_file',
'transitions',
'all_ref_transitions',
'ref_transition',
'max_torque',
'bang_action',
'old_net_update_delay',
'min_train_gap',
'min_action_gap',
'final_epsilon',
'epsilon_anneal_time',
'no_op_time',
'minibatch_size',
'gamma',
'sim_dt',
'start_epsilon',
'epsilon',
'last_old_net_update_count',
'net_update_count',
'last_train_time',
'last_action_time',
'ready_to_train',
'time_step',
'times_trained',
'last_state',
'last_action',
'train_t',
'sim_start_time',
'sim_num',
'a_hists',
'current_a_his',
's_hists',
'current_s_his',
'mse_hist',
'mse_hist2',
'rmse_rel_hist',
'rmse_rel_hist2',
'load_path',
'current_net',
'old_net',
'transfer_path',
'save_path',
]

to_replace = sorted(to_replace, lambda x,y: len(y) - len(x))

for i in range(len(to_replace)):
    s = s.replace(to_replace[i], 'TTT'+str(i)+'TTT')

for i in range(len(to_replace)):
    s = s.replace('TTT'+str(i)+'TTT', 'self.' + to_replace[i])

print s
