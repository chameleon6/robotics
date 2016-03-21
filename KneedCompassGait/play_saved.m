function play_saved(model_num)

% options = [];
% options.floating = true;
% options.terrain = RigidBodyFlatTerrain();
% options.twoD = true;
% options.view = 'right';
% %m = PlanarRigidBodyManipulator('KneedCompassGait.urdf', options);
% %r = TimeSteppingRigidBodyManipulator(m,.001);
% r = TimeSteppingRigidBodyManipulator('KneedCompassGait.urdf', 0.001, options);
% c = SNController(r, true, model_num);
% sys = feedback(r,c);
%
% v = r.constructVisualizer;
% v.axis = [-1.0 8.0 -0.1 2.1];
% v.display_dt = 0.05;

%xtraj = simulate(sys, [0 0.005]);

d = load(sprintf('trajs/%d.mat', model_num))
xtraj = d.xtraj
v = d.v

p_opts = struct('slider', true);
v.playback(xtraj, p_opts);

end
