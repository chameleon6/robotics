classdef PlaybackController < DrakeSystem

  properties
    p
    controls
    states
    matlab_state_file
    python_action_file
  end

  methods
    function obj = PlaybackController(plant, controls, states)

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

      obj = obj@DrakeSystem(0,1,18,6,true,false);

      obj.controls = controls;
      obj.states = states;

      obj.matlab_state_file = strcat(pwd,'/../NN/matlab_state_file.out');
      obj.python_action_file = strcat(pwd,'/../NN/python_action_file.out');
      obj.p = plant;
      obj = obj.setInputFrame(plant.getStateFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
    end

    function x0 = getInitialState(obj)
      x0 = [2; 0];
    end

    function h = foot_height(obj, base_z, hip_angle, knee_angle)
      h = base_z - 0.5*(cos(hip_angle) + cos(hip_angle + knee_angle));
    end

    function h = left_foot_height(obj,x)
      base_z = x(2);
      base_relative_pitch = x(3);
      left_knee_pin = x(5);

      h = obj.foot_height(base_z, base_relative_pitch, left_knee_pin);
    end

    function h = right_foot_height(obj,x)

      base_z = x(2);
      base_relative_pitch = x(3);
      hip_pin = x(7);
      right_knee_pin = x(8);

      h = obj.foot_height(base_z, base_relative_pitch+hip_pin, right_knee_pin);
    end

    function y = update(obj,t,old_y,x)
      %fprintf('update')
      %old_y
      %x
      %t
      y = old_y+1;
    end

    %function ts = getSampleTime(obj)
    %  ts = [0.001 0; 0 0];
    %end

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
      %fprintf('output')
      %x
      %t
      ot = y;

      %if t > 0.001
      %  error('done')
      %end

      if mod(ot, 100) == 2 | t < 0.005
        ot
        pred = obj.states(:, ot)
        actual = x
        diff = pred - actual
      end

      ot = ot + 1;
      u = obj.controls(:, ot-1);
    end
  end

end
