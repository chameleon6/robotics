cd ~/drake-distro/drake/examples/KneedCompassGait
global state_targets;
start_time = cputime;

options = [];
options.floating = true;
options.terrain = RigidBodyFlatTerrain();
options.twoD = true;
options.view = 'right';
%m = PlanarRigidBodyManipulator('KneedCompassGait.urdf', options);
%r = TimeSteppingRigidBodyManipulator(m,.001);
r = TimeSteppingRigidBodyManipulator('KneedCompassGait.urdf', 0.001, options);
good_out_file = fopen('good_simbicon_output.out', 'a');
all_out_file = fopen('all_simbicon_output.out', 'a');

%q = zeros(3,1);
%qd = zeros(3,1);
%[H,C,B,dH,dC,dB] = manipulatorDynamics(r,q,qd);
v = r.constructVisualizer;
v.axis = [-1.0 8.0 -0.1 2.1];

v.display_dt = .05;
sim_len = 0.5;

good_sim_count = 0;
trajectories = []
traj_count = 1

for i = 1:1

  clk = clock;
  model_num = round(clk(6)*1000000)
  c = SNController(r, true, model_num);
  %c = SNController(r);
  sys = feedback(r,c);

  x0 = Point(sys.getStateFrame());

  start_state = 3
  start_pose = state_targets{start_state} + 0.2 * ones(6,1); %0.05 * randn(6,1);
  %start_pose = [0.0, 0.25, 0, 0]

  x0.torso_pin = start_pose(1);
  x0.hip_pin = start_pose(2);
  x0.left_knee_pin = start_pose(3);
  x0.right_knee_pin = start_pose(4);
  x0.left_ankle_pin = start_pose(5);
  x0.right_ankle_pin = start_pose(6);
  x0.base_relative_pitch = 0.1;
  x0.base_z = 1.04;
  x0.base_xdot = 0.3;
  %x0.x1 = 4; %start_state

  xtraj = simulate(sys, [0 sim_len], x0);
  runtime = cputime - start_time

  fprintf(all_out_file, '%s\n', c.out_file_name);
  save(sprintf('trajs/%d.mat', model_num), 'xtraj', 'v')
  x_f = xtraj.eval(sim_len)

  if x_f(2) > 0.9
    fprintf(good_out_file, '%s\n', c.out_file_name);
    good_sim_count = good_sim_count + 1
    trajectories{traj_count} = xtraj;
    traj_count = traj_count + 1;
  end

  p_opts = struct('slider', true);
  v.playback(xtraj, p_opts);

end

fclose(good_out_file);
fclose(all_out_file);
