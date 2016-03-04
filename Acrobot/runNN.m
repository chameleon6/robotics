tic

pd = AcrobotPlant;
pv = AcrobotVisualizer(pd);
c = NNAcrobotController(pd);
c = setSampleTime(c, [0.01;0]);
sys = feedback(pd,c);
%sim_durations = [5];
sim_durations = [1 1 1 1 5 5 5 5 5 5 10 10 10];
num_train_sims = 8;
options = [];
options.slider = true;

trajectories = []
traj_count = 1

for i = sim_durations
  fprintf('sim %d duration %.2f\n', traj_count, i);
  start_state = randn(4,1);
  if traj_count <= num_train_sims
    start_state = [pi;0;0;0]+0.05*randn(4,1);
  end
  traj=simulate(sys,[0 i], start_state);
  trajectories{traj_count} = traj;
  if i > 1
    pv.playback(trajectories{traj_count}, options);
  end
  traj_count = traj_count + 1;
  %break;
end

%pv.playback(xtraj, options);

toc
