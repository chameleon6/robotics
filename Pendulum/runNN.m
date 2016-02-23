tic

pd = PendulumPlant;
pv = PendulumVisualizer();
c = NNController(pd); %balanceLQR(pd);
c = setSampleTime(c, [0.01;0]);
sys = feedback(pd,c);
sim_durations = [0.5 0.5 0.5 0.5 0.5 0.5 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 2 2 1 1 1 1 2 2 3 3 1 1 1 1 1 2 2 3];
options = [];
options.slider = true;

trajectories = []
traj_count = 1

for i = sim_durations
  fprintf('sim %d duration %.2f\n', traj_count, i);
  start_state = [pi;0]+1*randn(2,1);
  traj=simulate(sys,[0 10], start_state);
  trajectories{traj_count} = traj;
  pv.playback(trajectories{traj_count}, options);
  traj_count = traj_count + 1;
  break;
end

%pv.playback(xtraj, options);

toc
