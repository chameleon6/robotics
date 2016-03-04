classdef AcrobotController < DrakeSystem
  properties
    p
    A
    B
    Q
    R
    K
    S
    E_d
    k_1
    k_2
    k_3
    switching_thresh
  end

  methods
    function obj = AcrobotController(plant)
      obj = obj@DrakeSystem(0,0,4,1,true,true);
      obj.p = plant;
      obj = obj.setInputFrame(plant.getStateFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
      [f, df] = obj.p.dynamics(0, [pi;0;0;0], [0]);
      obj.A = df(:,2:5);
      obj.B = df(:,6);
      obj.Q = 10*eye(4);
      obj.R = eye(1);
      [obj.K,obj.S] = lqr(obj.A, obj.B, obj.Q, obj.R);

      com_position = obj.p.getCOM([pi;0]);
      mass = obj.p.getMass();
      gravity = obj.p.gravity;
      obj.E_d = -com_position(2) * mass * gravity(3);
      obj.k_1 = 1.0;
      obj.k_2 = 1.0;
      obj.k_3 = 2 * sqrt(obj.k_2);
      obj.switching_thresh = 3.0e3;

      global E_vs_t_history
      E_vs_t_history = [];
    end

    function u = output(obj,t,~,x)
      q = x(1:2);
      qd = x(3:4);

      % unwrap angles q(1) to [0,2pi] and q(2) to [-pi,pi]
      q(1) = q(1) - 2*pi*floor(q(1)/(2*pi));
      q(2) = q(2) - 2*pi*floor((q(2) + pi)/(2*pi));

      %%%% put your controller here %%%%
      % You might find some of the following functions useful
      [H,C,B] = obj.p.manipulatorDynamics(q,qd);
      com_position = obj.p.getCOM(q);
      mass = obj.p.getMass();
      gravity = obj.p.gravity;

      % Recall that the kinetic energy for a manipulator given by .5*qd'*H*qd
      E = -com_position(2) * mass * gravity(3) + 0.5 * qd' * H * qd;
      u_e = obj.k_1 * (obj.E_d - E) * qd(2);

      y = -obj.k_2 * q(2) - obj.k_3 * qd(2);
      u_p = C(2) + H(1,2) * (H(1,2)*y - C(1)) / H(1,1) + H(2,2)*y;

      x0 = [pi;0;0;0];
      x = [q; qd];
      cost_to_go = (x - x0)' * obj.S * (x - x0);

      if cost_to_go > obj.switching_thresh
        % PFL
        u = u_e + u_p;
      else
        % LQR
        u = -obj.K*(x - [pi;0;0;0]);
      end

      global E_vs_t_history
      E_vs_t_history = [E_vs_t_history [t; E]];

      %%%% end of your controller %%%%

      % leave this line below, it limits the control input to [-20,20]
      u = max(min(u,20),-20);
      % This is the end of the function
    end
  end

  methods (Static)
    function [t,x]=run()
      plant = PlanarRigidBodyManipulator('Acrobot.urdf');
      controller = AcrobotController(plant);
      v = plant.constructVisualizer;
      sys_closedloop = feedback(plant,controller);

      x0 = [.1*(rand(4,1) - 1)]; % start near the downward position
      %x0 = [pi - .5*randn;0;0;0];  % start near the upright position
      %x0 = [pi-0.1;0.15;0;0.5];  % use this for problem 3 (b)
      global xtraj
      options = [];
      options.slider = true;
      xtraj=simulate(sys_closedloop,[0 5],x0);
      v.axis = [-4 4 -4 4];
      playback(v,xtraj, options);
      t = xtraj.pp.breaks;
      x = xtraj.eval(t);
      %x(1:2,:) = mod(x(1:2,:), 2*pi);

      % Energy vs time plotting code
      global E_vs_t_history;
      [~,inds] = sort(E_vs_t_history(1,:));
      %E_vs_t_history = E_vs_t_history(:, inds);
      figure();
      hold on;
      plot([0 5], [controller.E_d controller.E_d], 'g-');
      plot(E_vs_t_history(1,:), E_vs_t_history(2,:));
      hold off;
      xlabel('t');
      ylabel('E');
      title(sprintf('k_1=%.2f', controller.k_1));

      % Phase space plotting
      figure(11);
      subplot(1,2,1)
      plot(x(1,:),x(3,:),'r-','LineWidth',2);
      hold on;
      plot(pi,0,'g.','LineWidth',10);
      plot(3*pi,0,'g.','LineWidth',10);
      plot(-pi,0,'g.','LineWidth',10);
      plot(x(1,1),x(3,1),'b.','LineWidth',10);
      hold off;
      xlabel('theta 1');
      ylabel('theta 1 dot');

      subplot(1,2,2)
      plot(x(2,:),x(4,:),'r-','LineWidth',2);
      hold on;
      plot(0,0,'g.','LineWidth',10);
      plot(2*pi,0,'g.','LineWidth',10);
      plot(-2*pi,0,'g.','LineWidth',10);
      plot(x(2,1),x(4,1),'b.','LineWidth',10);
      hold off;
      xlabel('theta 2');
      ylabel('theta 2 dot');

    end
  end
end
