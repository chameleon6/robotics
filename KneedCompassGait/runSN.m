cd ~/drake-distro/drake/examples/KneedCompassGait
clear;
global sim_fail_time;
global sim_failed;
global state_targets;
global current_target_state;
global last_reward_x_step;
global lowest_ground_so_far;
global failed_current_ground_h;
global rewarded_current_ground_h;
global ff_torques;
load('ff.mat')
ff_torques = zeros(size(ff_torques));

start_time = cputime;

options = [];
options.floating = true;

%%%
%box_xs = [-1];
%box_h = -1;
%options.terrain = RigidBodyFlatTerrain();
%%%

%%%options.terrain = RigidBodyStepTerrain(boxes);

options.twoD = true;
options.view = 'right';

%good_out_file = fopen('outputs/good_simbicon_files.out', 'a');
%all_out_file = fopen('outputs/all_simbicon_files.out', 'a');

num_box_ft = 3;
v.display_dt = .001;
sim_len = 10;
good_sim_count = 0;
trajectories = [];
visualizers = [];
traj_count = 1;
model_nums = [];
good_traj_inds = [];
box_lists = [];
num_failed = 0;
p_opts = struct('slider', true);

for i = 1:1

  fprintf('sim %d\n', i);
  fprintf('good_traj_count %d\n', size(good_traj_inds, 2));
  fprintf('num_failed %d\n', num_failed);

  tic

  r = TimeSteppingRigidBodyManipulator('KneedCompassGait.urdf', 0.001, options);

  %%
  box_xs = [-1; unifrnd(0.2, 0.4)];
  num_boxes = 15;
  for i2 = 1:num_boxes
    %box_xs = [box_xs; box_xs(end) + unifrnd(0.8, 1)];
    box_xs = [box_xs; box_xs(end) + unifrnd(0.6, 0.8)];
  end

  % flat ground
  num_boxes = 4;
  box_xs = [-1; 100; 200; 300; 400];

  box_h = 0.06;
  arg_str = sprintf('%f ', [box_h; box_xs]);
  command_str = sprintf('python make_box_urdf.py %s', arg_str);
  system(command_str);
  box_lists{traj_count} = box_xs;
  %boxes = make_boxes(box_xs, box_h);
  r = r.addRobotFromURDF('pybox.urdf');
  %%%

  v = r.constructVisualizer;
  v.axis = [-1.0 10.0 -0.1 3.5];

  sim_fail_time = inf;
  sim_failed = false;
  lowest_ground_so_far = -1;
  last_reward_x_step = 0;
  failed_current_ground_h = true;
  rewarded_current_ground_h = true;

  clk = clock;
  model_num = round(clk(6)*1000000)
  c = SNController(r, 0, model_num, 0.01, box_xs, box_h, num_box_ft, 0);
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

  x0.base_z = 1.9;
  x0.base_zdot = 0.0;
  x0.base_xdot = 0.5;
  x0.x1 = mod(start_state,4) + 1; %start_state
  current_target_state = x0.x1;
  x0(4:9) = start_pose;
  x0(1) = 0;
  x0(2) = x0(2) - min(c.left_foot_height(x0), c.right_foot_height(x0)) + 0.001; % base_z
  %x0 = c.reflect_state(x0)
  %x0(1:18) = [0.432819; 1.261991; 0.219682; -0.254988; 0.467295; 1.920353; -0.229881; 0.086108; 1.813502; 0.813278; -0.464504; 0.087726; -0.636299; 2.932976; 1.953149; 1.122909; -0.793436; -0.006064];
  %x0(10) = -5;
  %x0(1) = 0.5;


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

  success = false;
  try
    xtraj = simulate(sys, [0 sim_len], x0);
    success = true;
  catch
    fprintf('error in sim %d', i);
    num_failed = num_failed + 1;
  end

  runtime = cputime - start_time

  %fprintf(all_out_file, '%s\n', c.out_file_name);
  %x_f = xtraj.eval(sim_len);

  %fprintf(good_out_file, '%s\n', c.out_file_name);
  %good_sim_count = good_sim_count + 1
  if success
    trajectories{traj_count} = xtraj;
    visualizers{traj_count} = v;
    if ~sim_failed
      good_traj_inds = [good_traj_inds traj_count];
    end
    traj_count = traj_count + 1;
    model_nums = [model_nums model_num];
    %v.playback(xtraj, p_opts);
  end

  if size(good_traj_inds, 2) > 10
    break
  end

  toc

  fclose(c.out_file)
  % figure
  % hold on;
  % plot(log(1,:), 'r')
  % plot(log(2,:), 'r')
  % plot(log(3,:), 'b')
  % plot(log(4,:), 'g')

end

save('ff.mat', 'ff_torques')

%fclose(good_out_file);
%fclose(all_out_file);
