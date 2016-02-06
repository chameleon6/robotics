pd = PendulumPlant;
pv = PendulumVisualizer();
c = balanceLQR(pd);
sys = feedback(pd,c);
for i=1:1
  xtraj=simulate(sys,[0 1],[pi;0]+0.2*randn(2,1));
  pv.playback(xtraj);
end
