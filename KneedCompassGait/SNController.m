classdef SNController < DrakeSystem

  properties
    p
    out_file
    out_file_name
    use_net % 0 = simbicon, 1 = NN action, 2 = NN discrete
    matlab_state_file
    python_action_file
    output_dt
    reward_x_step
  end

  methods
    function obj = SNController(plant, use_net, model_num, output_dt)

      global state_targets;
      torso_lean = 0.1;
      max_hip_angle = 0.7;
      max_knee_angle = 0.7;
      leg_cross = 0.6;
      straight_knee = 0.1;
      bend_ankle = pi/2 + 0.3;
      kick_ankle = pi/2 + 0.5;

      %torso_lean = 0.1;
      %max_hip_angle = 1.0;
      %max_knee_angle = 1.0;
      %leg_cross = 1.0;
      %straight_knee = 0.1;
      %bend_ankle = pi/2 + 0.3;
      %kick_ankle = pi/2 + 0.5;

      % torso, hip, left_knee, right_knee, left_ankle, right_ankle
      % state_targets = {
      %   [torso_lean; -leg_cross; 0; max_knee_angle], % right bend
      %   [torso_lean; -max_hip_angle; 0; 0], % right straight in front
      %   [torso_lean; leg_cross; max_knee_angle; 0], % left bend
      %   [torso_lean; max_hip_angle; 0; 0] % left straight in front
      % };

      state_targets = {
        [leg_cross + torso_lean; leg_cross; max_knee_angle; straight_knee; bend_ankle; bend_ankle], % left bend
        [-max_hip_angle + torso_lean; -max_hip_angle; straight_knee; straight_knee; kick_ankle; kick_ankle], % left kick back
        [torso_lean; -leg_cross; straight_knee; max_knee_angle; bend_ankle; bend_ankle], % right bend
        [torso_lean; max_hip_angle; straight_knee; straight_knee; kick_ankle; kick_ankle] % right kick back
      };

      obj = obj@DrakeSystem(0,2,18,6,true,false);
      % y(3) = last_rewarded x_step

      obj.reward_x_step = 0.2;
      obj.matlab_state_file = strcat(pwd,'/../NN/matlab_state_file.out');
      obj.python_action_file = strcat(pwd,'/../NN/python_action_file.out');
      obj.p = plant;
      obj = obj.setInputFrame(plant.getStateFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
      c = clock;
      obj.out_file_name = sprintf('outputs/%d.out', model_num);
      obj.out_file = fopen(obj.out_file_name, 'a');
      obj.use_net = use_net;
      obj.output_dt = output_dt;
    end

    function x0 = getInitialState(obj)
      x0 = [2; 0];
    end

    function [h, rel_x] = foot_coords(obj, base_z, hip_angle, knee_angle)
      h = base_z - 0.5*(cos(hip_angle) + cos(hip_angle + knee_angle));
      rel_x = -0.5*(sin(hip_angle) + sin(hip_angle + knee_angle));
    end

    function [h, rel_x] = left_foot_coords(obj,x)
      base_z = x(2);
      base_relative_pitch = x(3);
      left_knee_pin = x(5);

      [h, rel_x] = obj.foot_coords(base_z, base_relative_pitch, left_knee_pin);
    end

    function h = left_foot_height(obj, x)
      [h, ~] = obj.left_foot_coords(x);
    end

    function [h, rel_x] = right_foot_coords(obj,x)

      base_z = x(2);
      base_relative_pitch = x(3);
      hip_pin = x(7);
      right_knee_pin = x(8);

      [h, rel_x] = obj.foot_coords(base_z, base_relative_pitch+hip_pin, right_knee_pin);
    end

    function h = right_foot_height(obj, x)
      [h, ~] = obj.right_foot_coords(x);
    end



    function y = update(obj,t,old_y,x)

      if obj.use_net == 0
        state = old_y(1);
        last_update_time = old_y(2);
        update_interval = 0.4;

        %base_z = x(2);
        %base_relative_pitch = x(3);
        %left_knee_pin = x(5);
        %hip_pin = x(7);
        %right_knee_pin = x(8);

        %left_h = foot_height(base_relative_pitch, left_knee_pin);
        %right_h = foot_height(base_relative_pitch+hip_pin, right_knee_pin);

        time_up = t - last_update_time > update_interval;
        [left_h, left_x] = obj.left_foot_coords(x);
        [right_h, right_x] = obj.right_foot_coords(x);

        should_update = false;
        if state == 1 || state == 3
          should_update = time_up;
        elseif state == 2
          should_update = left_h < 0.0005;
        else
          should_update = right_h < 0.0005;
        end

        if should_update
          t
          state = mod(state+1, 4)
          if state == 0
            state = 4
          end
          last_update_time = t;
        end

        y = [state; last_update_time];
      else
        y = [0;0];
      end
    end

    %function ts = getSampleTime(obj)
    %  ts = [0.001 0; 0 0];
    %end

    function [r, term] = reward(obj,x,t)

      global sim_failed;
      global sim_fail_time;
      global last_reward_x_step;
      global log;
      [left_h, left_x] = obj.left_foot_coords(x);
      [right_h, right_x] = obj.right_foot_coords(x);

      [c,J] = obj.p.getCOM(x);
      qd = x(10:end);
      px = sqrt(c(2)/9.81)*J*qd;
      px = px(1);
      %log = [log [(left_x+right_x)/2; x(10)-0.5; c(1)-x(1)]];
      %log = [log x(10)];

      if x(2) > 0.9 & x(2) < 1.05 & x(10) > 0
        num_x_steps = floor(x(1)/obj.reward_x_step);
        if num_x_steps > last_reward_x_step & left_x * right_x < 0 & x(10) > 0.5
          last_reward_x_step = num_x_steps;
          t
          r = 1
          x
        else
          r = 0;
        end

        %r = -(c(1) - (left_x + right_x)/2)^2;

        %r1 = -(x(10)-0.5)^2
        %r2 = -(left_x+right_x)^2
        %r = r1+r2

        term = 0;
      else
        if ~sim_failed
          sim_failed = true;
          sim_fail_time = t;
          %r = -10;
          r = 0;
          term = 1;
        end
      end
    end

    function x_new = transform_state(obj, x)
      [left_h, left_x] = obj.left_foot_coords(x);
      left_contact = left_h < 0.0005;
      [right_h, right_x] = obj.right_foot_coords(x);
      right_contact = right_h < 0.0005;
      x_new = [x(2:end); left_contact; right_contact; left_x; right_x];
    end

    function write_state(obj,x,t)

      x_new = obj.transform_state(x);
      f = fopen(obj.matlab_state_file, 'w');
      %x_new(1) = mod(x(1), 2*pi);
      [r, term] = obj.reward(x,t);
      fprintf(f, '%f\n', r);
      fprintf(f, '%d\n', term);
      fprintf(f, '%d ', x_new);
      fprintf(f, '\n');
      fprintf(f, '%d\n', t);
      fclose(f);

      % %debug
      % fprintf('writing state\n');
      % fprintf('%d\n', obj.reward(x));
      % fprintf('%d ', x);
      % fprintf('\n');
    end

    function a = get_action(obj)
      start_time = cputime;
      while true
        while exist(obj.python_action_file, 'file') ~= 2
          if cputime - start_time > 10
            cputime - start_time
            error('timeout')
          end
          continue;
        end

        f = fopen(obj.python_action_file, 'r');
        a = fscanf(f, '%f ');
        fclose(f);
        if isempty(a)
          fprintf('python incomplete output\n')
          if cputime - start_time > 10
            cputime - start_time
            error('timeout')
          end
          continue;
        end

        delete(obj.python_action_file);

        %debug
        % fprintf('read state:\n');
        % fprintf('%d ', a);
        % fprintf('\n');
        break;
      end

    end

    function u = output(obj,t,y,x)

      % if mod(int16(t*100),5) == 0
      %   t
      %   y
      %   coordinates = obj.p.getStateFrame().getCoordinateNames();
      %   for i=1:length(coordinates)
      %     fprintf(1,'%20s = %f\n',coordinates{i},x(i));
      %   end
      % end

      %left_foot = obj.left_foot_height(x)
      %right_foot = obj.right_foot_height(x)

      global sim_fail_time
      if t - sim_fail_time > 0.0 | t == 0
        u = zeros(6,1);
        return
      end

      %if t > 0.01
      %  sim_failed = true;
      %end

      %if t < 0.01
      %  u = [0;0;0;0;0;0];
      %  return;
      %end

      if obj.use_net == 1
        obj.write_state(x,t);
        u = obj.get_action();
        if mod(int16(t*1000), 10) == 0
          t, x, u;
        end
      else

        global current_target_state
        num_dts = t/obj.output_dt;

        if abs(num_dts - round(num_dts)) < 0.00001
          state_ind = -1;
          if obj.use_net == 0
            state_ind = y(1);
            [r, term] = obj.reward(x,t);
            fprintf(obj.out_file, '%f ', r);
            fprintf(obj.out_file, '\n');
            fprintf(obj.out_file, '%.10f ', obj.transform_state(x));
            fprintf(obj.out_file, '\n');
            fprintf(obj.out_file, '%d ', state_ind - 1);
            fprintf(obj.out_file, '\n');
          else %use_net == 2
            obj.write_state(x,t);
            state_ind = round(obj.get_action()) + 1;
          end
          current_target_state = state_ind;

        end

        global torso_lean;
        global max_hip_angle;
        global max_knee_angle;
        global state_targets;

        %torso_lean_rel = torso_lean - x(2);
        targets = state_targets{current_target_state};

        % torso, hip, left_knee, right_knee, left_ankle, right_ankle
        joint_inds = [4;7;5;8;6;9];
        joint_vel_inds = 9 + joint_inds;
        actuals = x(joint_inds);
        vels = x(joint_vel_inds);
        p_const = 200;
        alphas_p = p_const*[0.1; 1; 1; 1; .1; .1];
        alphas_d = 2*sqrt(alphas_p); %d_const*[10; 1; 1; 1; 1; 1];
        u = alphas_p .* (targets - actuals) - alphas_d .* vels;
        % u = u + 3*randn(6,1);
        u = min(max(u, -50),50);
      end


      %u = Point(obj.p.getInputFrame())
      %[H,C,B] = obj.p.manipulatorDynamics(x(1:2),x(3:4));
      %l = obj.p.l1+obj.p.l2;
      %b = .1;
      %u = C + H*[-obj.p.g*sin(x(1))/l - b*x(3);0];
    end
  end

end
