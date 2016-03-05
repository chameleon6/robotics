cd ~/drake-distro/drake/examples/KneedCompassGait
global state_targets;
start_time = cputime;

options = [];
options.floating = true;
options.terrain = RigidBodyFlatTerrain();
options.twoD = true;
options.view = 'right';
r = TimeSteppingRigidBodyManipulator('KneedCompassGait.urdf', 0.001, options);

v = r.constructVisualizer;
v.axis = [-1.0 8.0 -0.1 2.1];

v.display_dt = .05;
sim_len = 2;

good_sim_count = 0;
trajectories = []
traj_count = 1

c = SimbiconController(r);
sys = feedback(r,c);

x0 = Point(sys.getStateFrame());

start_state = 3
start_pose = state_targets{start_state} + 0.1 * ones(6,1);

x0.torso_pin = start_pose(1);
x0.hip_pin = start_pose(2);
x0.left_knee_pin = start_pose(3);
x0.right_knee_pin = start_pose(4);
x0.left_ankle_pin = start_pose(5);
x0.right_ankle_pin = start_pose(6);
x0.base_relative_pitch = 0.1;
x0.base_z = 1.04;
x0.base_xdot = 0.3;
x0.x1 = 4; %start_state

% Run simulation, then play it back at realtime speed
xtraj = simulate(sys, [0 sim_len], x0);
runtime = cputime - start_time
p_opts = struct('slider', true);
v.playback(xtraj, p_opts);
