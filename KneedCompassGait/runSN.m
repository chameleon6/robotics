cd ~/drake-distro/drake/examples/KneedCompassGait
global sim_fail_time;
global sim_failed;
global state_targets;
global current_target_state;
global last_reward_x_step;
start_time = cputime;

options = [];
options.floating = true;
options.terrain = RigidBodyFlatTerrain();
%options.terrain = RigidBodyStepTerrain([0.5 0 0.25 1000 0.05]);

options.twoD = true;
options.view = 'right';
%m = PlanarRigidBodyManipulator('KneedCompassGait.urdf', options);
%r = TimeSteppingRigidBodyManipulator(m,.001);
r = TimeSteppingRigidBodyManipulator('KneedCompassGait.urdf', 0.001, options);
good_out_file = fopen('outputs/good_simbicon_files.out', 'a');
all_out_file = fopen('outputs/all_simbicon_files.out', 'a');

%q = zeros(3,1);
%qd = zeros(3,1);
%[H,C,B,dH,dC,dB] = manipulatorDynamics(r,q,qd);
v = r.constructVisualizer;
v.axis = [-1.0 8.0 -0.1 2.1];

v.display_dt = .05;
sim_len = 3.0;

good_sim_count = 0;
trajectories = [];
traj_count = 1;
model_nums = [];
good_traj_inds = [];

for i = 1:30

  tic

  sim_fail_time = inf;
  sim_failed = false;
  last_reward_x_step = 0;

  clk = clock;
  model_num = round(clk(6)*1000000);
  c = SNController(r, 2, model_num, 0.01);
  c = setSampleTime(c, [0.001;0]);
  %c = SNController(r);
  sys = feedback(r,c);

  x0 = Point(sys.getStateFrame());

  if mod(i, 2) == 0
    start_state = 3
  else
    start_state = 1
  end

  start_pose = state_targets{start_state}; % + 0.01 * randn(6,1);
  %start_pose = [0.0, 0.25, 0, 0]

  x0.torso_pin = start_pose(1);
  x0.hip_pin = start_pose(2);
  x0.left_knee_pin = start_pose(3);
  x0.right_knee_pin = start_pose(4);
  x0.left_ankle_pin = start_pose(5);
  x0.right_ankle_pin = start_pose(6);
  if start_state == 1
    x0.base_relative_pitch = -0.5;
  else
    x0.base_relative_pitch = 0.1;
  end
  x0.base_z = 1.04;
  x0.base_zdot = 0.0;
  x0.base_xdot = 0.3;
  x0.x1 = mod(start_state,4) + 1; %start_state
  x0.base_z = x0.base_z - min(c.left_foot_height(x0), c.right_foot_height(x0)) + 0.01

  current_target_state = x0.x1;

  %x0.torso_pin = start_pose(1);
  %x0.hip_pin = start_pose(2);
  %x0.left_knee_pin = start_pose(3);
  %x0.right_knee_pin = start_pose(4);
  %x0.left_ankle_pin = start_pose(5);
  %x0.right_ankle_pin = start_pose(6);
  %x0.base_relative_pitch = 0.1;
  %x0.base_z = 1.04;
  %x0.base_zdot = 0.0;
  %x0.base_xdot = 1.0;
  %x0.x1 = 4; %start_state
  %x0.base_z = x0.base_z - min(c.left_foot_height(x0), c.right_foot_height(x0)) + 0.01;


  xtraj = simulate(sys, [0 sim_len], x0);
  runtime = cputime - start_time

  fprintf(all_out_file, '%s\n', c.out_file_name);
  x_f = xtraj.eval(sim_len)

  fprintf(good_out_file, '%s\n', c.out_file_name);
  %good_sim_count = good_sim_count + 1
  trajectories{traj_count} = xtraj;
  traj_count = traj_count + 1;
  %end

  toc

  if x_f(2) > 0.9
    model_nums = [model_nums model_num];
  end

  p_opts = struct('slider', true);
  %v.playback(xtraj, p_opts);
  if sim_fail_time > 2.0
    good_traj_inds = [good_traj_inds i];
  end

end

fclose(good_out_file);
fclose(all_out_file);
