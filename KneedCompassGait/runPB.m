cd ~/drake-distro/drake/examples/KneedCompassGait
global state_targets;
start_time = cputime;

options = [];
options.floating = true;
%options.terrain = RigidBodyFlatTerrain();
options.terrain = RigidBodyStepTerrain([1 0 1 1000 0.05]);
options.twoD = true;
options.view = 'right';
%m = PlanarRigidBodyManipulator('KneedCompassGait.urdf', options);
%r = TimeSteppingRigidBodyManipulator(m,.001);
r = TimeSteppingRigidBodyManipulator('KneedCompassGait.urdf', 0.001, options);
good_out_file = fopen('good_simbicon_output.out', 'a');
all_out_file = fopen('all_simbicon_output.out', 'a');

v = r.constructVisualizer;
v.axis = [-1.0 8.0 -0.1 2.1];

v.display_dt = .05;
sim_len = 0.01;

good_sim_count = 0;
trajectories = []
traj_count = 1

%model_num = 298090
model_num = 532468
%model_num = 11111
f = fopen(sprintf('old_outputs/%d.out', model_num), 'r');

for i = 1:1

  controls = zeros(6, 0);
  states = zeros(18, 0);
  while true
    l = fgetl(f);
    if l == -1
      break
    end
    l2 = fgetl(f);
    state = sscanf(l, '%f ');
    control = sscanf(l2, '%f ');
    states = [states state];
    controls = [controls control];
  end

  %c = PlaybackController(r, controls, states);
  c = SNController(r, false, 11111);
  c = setSampleTime(c, [0.001;0]);

  sys = feedback(r,c);

  % x0 = Point(sys.getStateFrame());

  % start_state = 3
  % start_pose = state_targets{start_state} + 0.2 * ones(6,1); %0.05 * randn(6,1);
  % %start_pose = [0.0, 0.25, 0, 0]

  % x0.torso_pin = start_pose(1);
  % x0.hip_pin = start_pose(2);
  % x0.left_knee_pin = start_pose(3);
  % x0.right_knee_pin = start_pose(4);
  % x0.left_ankle_pin = start_pose(5);
  % x0.right_ankle_pin = start_pose(6);
  % x0.base_relative_pitch = 0.1;
  % x0.base_z = 1.04;
  % x0.base_zdot = 0.0;
  % x0.base_xdot = 0.3;
  % x0.x1 = 4; %start_state
  % x0.base_z = x0.base_z - min(c.left_foot_height(x0), c.right_foot_height(x0))

  x0 = [states(:,1); 4; 0];
  %x0 = [states(:,1); 1]
  xtraj = simulate(sys, [0 sim_len], x0);
  runtime = cputime - start_time

  p_opts = struct('slider', true);
  v.playback(xtraj, p_opts);

end

fclose(good_out_file);
fclose(all_out_file);
