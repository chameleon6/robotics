classdef SNController < DrakeSystem

  properties
    p
    out_file
    out_file_name
    use_net
    matlab_state_file
    python_action_file
  end

  methods
    function obj = SNController(plant, use_net, model_num)

      global state_targets;
      torso_lean = 0.1;
      max_hip_angle = 0.7;
      max_knee_angle = 0.7;
      leg_cross = 0.6;
      straight_knee = 0.1;
      bend_ankle = pi/2 + 0.3;
      kick_ankle = pi/2 + 0.5;

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

      obj.matlab_state_file = strcat(pwd,'/../NN/matlab_state_file.out');
      obj.python_action_file = strcat(pwd,'/../NN/python_action_file.out');
      obj.p = plant;
      obj = obj.setInputFrame(plant.getStateFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
      c = clock;
      obj.out_file_name = sprintf('outputs/%d.out', model_num);
      obj.out_file = fopen(obj.out_file_name, 'w');
      obj.use_net = use_net
    end

    function x0 = getInitialState(obj)
      x0 = [2; 0];
    end

    function h = foot_height(hip_angle, knee_angle)
      h = base_z - 0.5*(cos(hip_angle) + cos(hip_angle + knee_angle));
    end

    function y = update(obj,t,old_y,x)

      if ~obj.use_net
        state = old_y(1);
        last_update_time = old_y(2);
        update_interval = 0.4;

        base_z = x(2);
        base_relative_pitch = x(3);
        left_knee_pin = x(5);
        hip_pin = x(7);
        right_knee_pin = x(8);

        time_up = t - last_update_time > update_interval;
        left_h = foot_height(base_relative_pitch, left_knee_pin);
        right_h = foot_height(base_relative_pitch+hip_pin, right_knee_pin);

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

    function ts = getSampleTime(obj)
      ts = [0.001 0; 0 0];
    end

    function r = reward(obj,x)
      r = x(2);
    end

    function write_state(obj,x,t)
      f = fopen(obj.matlab_state_file, 'w');
      x_new = x;
      x_new(1) = mod(x(1), 2*pi);
      r = obj.reward(x);
      fprintf(f, '%d\n', r);
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

      if t < 0.01
        u = [0;0;0;0;0;0];
      else
        if obj.use_net
          t, x
          obj.write_state(x,t);
          u = obj.get_action()
        else
          global torso_lean
          global max_hip_angle,
          global max_knee_angle
          global state_targets;
          torso_lean_rel = torso_lean - x(2);
          targets = state_targets{y(1)};

          % torso, hip, left_knee, right_knee, left_ankle, right_ankle
          joint_inds = [4;7;5;8;6;9];
          joint_vel_inds = 9 + joint_inds;
          actuals = x(joint_inds);
          vels = x(joint_vel_inds);
          p_const = 200;
          alphas_p = p_const*[1; 1; 1; 1; .1; .1];
          alphas_d = 2*sqrt(alphas_p); %d_const*[10; 1; 1; 1; 1; 1];
          u = alphas_p .* (targets - actuals) - alphas_d .* vels;
          u = u + 3*randn(6,1);
          u = min(max(u, -50),50);
        end

        if t > 0.01
          fprintf(obj.out_file, '%f ', x);
          fprintf(obj.out_file, '\n');
          fprintf(obj.out_file, '%f ', u);
          fprintf(obj.out_file, '\n');
        end
      end


      %u = Point(obj.p.getInputFrame())
      %[H,C,B] = obj.p.manipulatorDynamics(x(1:2),x(3:4));
      %l = obj.p.l1+obj.p.l2;
      %b = .1;
      %u = C + H*[-obj.p.g*sin(x(1))/l - b*x(3);0];
    end
  end

end
