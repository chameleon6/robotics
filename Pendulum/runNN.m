tic

global sim_failed;
pd = PendulumPlant;
pv = PendulumVisualizer();
c = NNController(pd); %balanceLQR(pd);
c = setSampleTime(c, [0.01;0]);
sys = feedback(pd,c);
%sim_durations = [2];
sim_durations = [4 4 4 4 4 4 4 4 4];
options = [];
options.slider = true;

trajectories = []
traj_count = 1

for i = sim_durations
  sim_failed = 0;
  fprintf('sim %d duration %.2f\n', traj_count, i);
  start_state = [pi/4;0]+0.1*randn(2,1);
  traj=simulate(sys, [0 i], start_state);
  trajectories{traj_count} = traj;
  pv.playback(trajectories{traj_count}, options);
  traj_count = traj_count + 1;
  %break;
end

%pv.playback(xtraj, options);

toc
