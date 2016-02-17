tic

pd = PendulumPlant;
pv = PendulumVisualizer();
c = NNController(pd); %balanceLQR(pd);
c = setSampleTime(c, [0.01;0]);
sys = feedback(pd,c);

for i = 1:8
  start_state = [pi;0] %[pi;0]+0.2*randn(2,1)
  xtraj=simulate(sys,[0 2], start_state);
  options = [];
  options.slider = true;
  pv.playback(xtraj, options);
end

toc
