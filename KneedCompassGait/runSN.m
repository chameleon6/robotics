cd ~/drake-distro/drake/examples/KneedCompassGait
global sim_fail_time;
global sim_failed;
global state_targets;
global current_target_state;
global last_reward_x_step;
global log;
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
v.axis = [-1.0 5.0 -0.1 2.1];

v.display_dt = .01;
sim_len = 2;
good_sim_count = 0;
trajectories = [];
traj_count = 1;
model_nums = [];
good_traj_inds = [];

for i = 1:1

  fprintf('sim %d\n', i);
  log = zeros(4,0);

  tic

  sim_fail_time = inf;
  sim_failed = false;
  last_reward_x_step = 0;

  clk = clock;
  model_num = round(clk(6)*1000000);
  c = SNController(r, 0, model_num, 0.1);
  c = setSampleTime(c, [0.001;0]);
  %c = SNController(r);
  sys = feedback(r,c);

  x0 = Point(sys.getStateFrame());

  if mod(i, 2) == 0
    start_state = 3;
  else
    start_state = 1;
  end

  start_pose = state_targets{start_state};% + 0.01 * randn(6,1);
  %start_pose = [0.0, 0.25, 0, 0]

  %x0.torso_pin = start_pose(1);
  %x0.hip_pin = start_pose(2);
  %x0.left_knee_pin = start_pose(3);
  %x0.right_knee_pin = start_pose(4);
  %x0.left_ankle_pin = start_pose(5);
  %x0.right_ankle_pin = start_pose(6);

  %if start_state == 1
  %  x0.base_relative_pitch = -0.5;
  %else
  %  x0.base_relative_pitch = 0.1;
  %end

  x0.base_z = 1.4;
  x0.base_zdot = 0.0;
  x0.base_xdot = 0.4;
  x0.x1 = mod(start_state,4) + 1; %start_state
  current_target_state = x0.x1;
  x0(4:9) = start_pose;
  x0(2) = x0(2) - min(c.left_foot_height(x0), c.right_foot_height(x0)) + 0.01; % base_z


  %if mod(i,2) == 1
  %  x0.x1 = 3;
  %  x0(1:18) = [0.0468806517 0.9790945205 -0.2809296345 0.4501658481 0.3531906305 1.8765274998 0.3031821861 0.2717239432 1.8953324087 0.5239818314 0.1212185094 0.9164383372 -1.7226071413 -0.7483170075 0.1319860563 -0.4583902179 -0.5088036654 -0.8196593879]';
  %else
  %  x0.x1 = 1;
  %  x0(1:18) = [0.0469637621 0.9796903126 0.0311596662 0.1102139609 0.2570784222 1.8951171387 -0.3085394053 0.3466207104 1.8773049595 0.5485918090 0.0983200806 0.4300624030 -1.2206228038 -0.2200556142 -0.9222081983 0.3696568784 -0.4707305204 0.1364917715]';
  %end

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
  x_f = xtraj.eval(sim_len);

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
  v.playback(xtraj, p_opts);
  if sim_fail_time > 2.0
    good_traj_inds = [good_traj_inds i];
  end


  % figure
  % hold on;
  % plot(log(1,:), 'r')
  % plot(log(2,:), 'r')
  % plot(log(3,:), 'b')
  % plot(log(4,:), 'g')

end

fclose(good_out_file);
fclose(all_out_file);
