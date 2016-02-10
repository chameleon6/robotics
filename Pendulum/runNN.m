tic

pd = PendulumPlant;
pv = PendulumVisualizer();
c = NNController(pd); %balanceLQR(pd);
sys = feedback(pd,c);

start_state = [pi/2;0] %[pi;0]+0.2*randn(2,1)
xtraj=simulate(sys,[0 10], start_state);
options = [];
options.slider = true;
pv.playback(xtraj, options);

toc
