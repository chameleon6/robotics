pd = PendulumPlant;
c = NNController(pd);

dt = 0.01;
t = 0;
xmin = -100;
xmax = 100;


num_refreshes = 0;

x = randn(1) * xmax;

for i = 1:1000
  %if mod(i, 100) == 0
  %  i,t
  %end
  c.write_state([0; x], t);
  u = c.get_action();
  x = x + u;
  t = t + dt;
  x
  if abs(x) > xmax
    x = randn(1) * xmax;
    num_refreshes = num_refreshes + 1
  end
end

