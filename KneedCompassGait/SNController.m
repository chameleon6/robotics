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
    n_boxes
    box_h
    box_xs
  end

  methods
    function obj = SNController(plant, use_net, model_num, output_dt, box_xs, box_h)

      % global state_targets;
      % torso_lean = 0.1;
      % max_hip_angle = 0.7;
      % max_knee_angle = 0.7;
      % leg_cross = 0.6;
      % straight_knee = 0.1;
      % bend_ankle = pi/2 + 0.3;
      % kick_ankle = pi/2 + 0.5;

      % % left leg, left knee, left ankle, right leg, right knee, right ankle
      % state_targets = {
      %   [-leg_cross/2 - torso_lean; max_knee_angle; bend_ankle; leg_cross/2 - torso_lean; straight_knee; bend_ankle], % left bend
      %   [max_hip_angle/2 - torso_lean; straight_knee; kick_ankle; -max_hip_angle/2 - torso_lean; straight_knee; kick_ankle], % left kick back
      %   [leg_cross/2 - torso_lean; straight_knee; bend_ankle; -leg_cross/2 - torso_lean; max_knee_angle; bend_ankle], % right bend
      %   [-max_hip_angle/2 - torso_lean; straight_knee; kick_ankle; max_hip_angle/2 - torso_lean; straight_knee; kick_ankle], % right kick back
      % };

      global state_targets;
      torso_lean = 0.;
      max_hip_angle = 1.4;
      max_knee_angle = 0.7;
      leg_cross = 1.2;
      straight_knee = 0.1;
      %bend_ankle = pi/2 + 0.3;
      %kick_ankle = pi/2 + 0.5;
      bend_ankle = pi/2 + 0.1;
      kick_ankle = pi/2;

      % left leg, left knee, left ankle, right leg, right knee, right ankle
      state_targets = {
        [-leg_cross/2 - torso_lean; max_knee_angle; bend_ankle; 0; straight_knee; bend_ankle], % left bend
        [max_hip_angle/2 - torso_lean; straight_knee; kick_ankle; 0; straight_knee; kick_ankle], % left kick back
        [0; straight_knee; bend_ankle; -leg_cross/2 - torso_lean; max_knee_angle; bend_ankle], % right bend
        [0; straight_knee; kick_ankle; max_hip_angle/2 - torso_lean; straight_knee; kick_ankle], % right kick back
      };

      obj = obj@DrakeSystem(0,2,18,6,true,false);
      % y(3) = last_rewarded x_step

      obj.n_boxes = size(box_xs, 1);
      obj.box_xs = box_xs;
      obj.box_h = box_h;
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

    function h = ground_h(obj, x)
      if obj.box_h < 0
        h = 0;
        return
      end

      for i = 1:obj.n_boxes
        if x + 0.001 < obj.box_xs(i)
          h = (obj.n_boxes - i + 1) * obj.box_h;
          return
        end
      end

      h = 0;
    end

    function x_drop = next_drop(obj, x)
      if obj.box_h < 0
        return
      end

      for i = 1:obj.n_boxes
        if x < obj.box_xs(i)
          x_drop = obj.box_xs(i);
          return
        end
      end
    end

    function state = feedback_adjust_state(obj, ind, x)
      global state_targets;
      v = x(10);
      c_v = 0.2;
      c_d = 2;
      state = state_targets{ind};
      if ind == 1
        state(1) = state(1) - c_v*v;
      elseif ind == 3
        state(4) = state(4) - c_v*v;
      elseif ind == 2
        [right_h, right_x] = obj.right_foot_coords(x);
        state(1) = state(1) - c_d * right_x;
      elseif ind == 4
        [left_h, left_x] = obj.left_foot_coords(x);
        state(4) = state(4) - c_d * left_x;
      end
    end

    function x = state_to_x(obj, ind)
      global state_targets;
      x = zeros(18,1);
      x(2) = 1.0;
      x(4:9) = state_targets{ind};
    end

    function x0 = getInitialState(obj)
      x0 = [2; 0];
    end

    function new_x = reflect_state(obj, x)
      new_x = x;
      new_x(4:6) = x(7:9);
      new_x(7:9) = x(4:6);
      new_x(13:15) = x(16:18);
      new_x(16:18) = x(13:15);

      %new_x(3) = x(3) + x(7); %base pitch
      %new_x(7) = -x(7); %hip
      %new_x(4) = x(4) - x(7);
      %new_x(13) = -x(13);
      %new_x(16) = -x(16);
      %new_x(5:6) = x(8:9);
      %new_x(8:9) = x(5:6);
      %new_x(14:15) = x(17:18);
      %new_x(17:18) = x(14:15);

    end

    function [h_rel_left, h_rel_right] = base_heights(obj, x)
      [left_h, left_x] = obj.left_foot_coords(x);
      [right_h, right_x] = obj.right_foot_coords(x);
      h_rel_left = x(2) - obj.ground_h(left_x + x(1));
      h_rel_right = x(2) - obj.ground_h(right_x + x(1));
    end

    function [h, rel_x] = foot_coords(obj, base_z, hip_angle, knee_angle)
      h = base_z - 0.5*(cos(hip_angle) + cos(hip_angle + knee_angle));
      rel_x = -0.5*(sin(hip_angle) + sin(hip_angle + knee_angle));
    end

    function [h, rel_x] = left_foot_coords(obj,x)
      base_z = x(2);
      base_relative_pitch = x(3);
      left_knee_pin = x(5);
      left_upper_leg_pin = x(4);

      [h, rel_x] = obj.foot_coords(base_z, left_upper_leg_pin + base_relative_pitch, left_knee_pin);
      h = h - obj.ground_h(rel_x + x(1));
    end

    function h = left_foot_height(obj, x)
      [h, ~] = obj.left_foot_coords(x);
    end

    function [h, rel_x] = right_foot_coords(obj,x)

      base_z = x(2);
      base_relative_pitch = x(3);
      right_upper_leg_pin = x(7);
      right_knee_pin = x(8);

      [h, rel_x] = obj.foot_coords(base_z, base_relative_pitch + right_upper_leg_pin, right_knee_pin);
      h = h - obj.ground_h(rel_x + x(1));
    end

    function h = right_foot_height(obj, x)
      [h, ~] = obj.right_foot_coords(x);
    end

    function b = on_box_edge(obj, foot_x, foot_h, t)
      b = false;
      if obj.box_h < 0
        return
      end

      for i = 1:obj.n_boxes
        if abs(obj.box_xs(i) - foot_x) < 0.01 & foot_h < obj.box_h
          b = true;
          %if abs(t*10 - round(t*10)) < 0.0001
          %  t
          %  foot_x
          %end
          return
        end
      end
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

        global lowest_ground_so_far;
        global failed_current_ground_h;
        global rewarded_current_ground_h;

        left_ground = obj.ground_h(left_x + x(1));
        right_ground = obj.ground_h(right_x + x(1));
        min_ground = min(left_ground, right_ground);

        if lowest_ground_so_far == -1
          if left_ground ~= right_ground
            error('did not start on equal ground')
          end
          lowest_ground_so_far = left_ground;
        end

        if min_ground < lowest_ground_so_far
          failed_current_ground_h = false;
          rewarded_current_ground_h = false;
          lowest_ground_so_far = min_ground;
        end

        left_on_edge = obj.on_box_edge(x(1) + left_x, left_h, t);
        right_on_edge = obj.on_box_edge(x(1) + right_x, right_h, t);
        if ((left_on_edge & left_ground == lowest_ground_so_far) | (right_on_edge & right_ground == lowest_ground_so_far)) & failed_current_ground_h == false
          t
          lowest_ground_so_far
          failed_current_ground_h = true
        end

        if state == 1 || state == 3
          should_update = time_up;
        elseif state == 2
          should_update = (left_h < 0.0005) | left_on_edge;
        else
          should_update = (right_h < 0.0005) | right_on_edge;
        end

        if should_update
          state = mod(state+1, 4);
          % left_h
          % right_h
          % left_x
          % right_x
          % left_x + x(1)
          % left_g = obj.ground_h(left_x+x(1))
          % right_g = obj.ground_h(right_x+x(1))
          if state == 0
            state = 4;
          end
          last_update_time = t;
          %t
          %state
        end

        y = [state; last_update_time];
      else
        y = [0;0];
      end
    end

    function [r, term] = reward(obj,x,t)

      global sim_failed;
      global sim_fail_time;
      global last_reward_x_step;
      global lowest_ground_so_far;
      global failed_current_ground_h;
      global rewarded_current_ground_h;

      [left_h, left_x] = obj.left_foot_coords(x);
      [right_h, right_x] = obj.right_foot_coords(x);
      left_ground = obj.ground_h(left_x + x(1));
      right_ground = obj.ground_h(right_x + x(1));

      [c,J] = obj.p.getCOM(x);
      qd = x(10:end);
      px = sqrt(c(2)/9.81)*J*qd;
      px = px(1);
      %log = [log [(left_x+right_x)/2; x(10)-0.5; c(1)-x(1)]];
      %log = [log x(10)];
      [bhl, bhr] = obj.base_heights(x);
      max_h = 1.5;
      min_h = 0.5;

      if ((bhl > min_h & bhl < max_h) | (bhr > min_h & bhr < max_h)) %& x(10) > 0
        %if left_x * right_x < 0
        %  r = 1;
        %else
        %  r = 0;
        %end
        epsilon = 0.001;
        if (left_ground == lowest_ground_so_far & left_h < epsilon) | (right_ground == lowest_ground_so_far & right_h < epsilon)
          if ~rewarded_current_ground_h & ~ failed_current_ground_h
            t
            r = 1
            rewarded_current_ground_h = true;
          else
            r = 0;
          end
        else
          r = 0;
        end

        term = 0;
      else
        if ~sim_failed
          sim_failed = true;
          sim_fail_time = t
          %r = -10;
          r = 0;
          term = 1;
        end
      end
    end

    function x_new = transform_state(obj, x, t)
      [left_h, left_x] = obj.left_foot_coords(x);
      left_contact = left_h < 0.0005;
      [right_h, right_x] = obj.right_foot_coords(x);
      right_contact = right_h < 0.0005;
      x_new = [x(2:end); left_contact; right_contact; left_x; right_x; left_h; right_h];
      if obj.box_h > 0
        lnd = obj.next_drop(left_x + x(1)) - (left_x + x(1));
        rnd = obj.next_drop(right_x + x(1)) - (right_x + x(1));
        cnd = obj.next_drop(x(1)) - x(1);
        x_new = [x_new; lnd; rnd; cnd];
      end

      %if abs(round(t*10) - t*10) < 0.00001
      %  t
      %  lnd
      %  rnd
      %  cnd
      %  left_h
      %  right_h
      %  left_x
      %  right_x
      %end
    end

    function write_state(obj,x,t)

      x_new = obj.transform_state(x, t);
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
          if cputime - start_time > 120
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

      %u = [0;0;0;0;0;0];
      %return

      global sim_fail_time
      global current_target_state

      if t - sim_fail_time > 0.0 | t == 0
        current_target_state = -1;
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

        num_dts = t/obj.output_dt;

        if abs(num_dts - round(num_dts)) < 0.00001 | current_target_state == -1
          % t
          % [rh, rx] = obj.right_foot_coords(x)
          % [lh, lx] = obj.left_foot_coords(x)
          state_ind = -1;
          if obj.use_net == 0
            state_ind = y(1);
            [r, term] = obj.reward(x,t);
            fprintf(obj.out_file, '%f ', r);
            fprintf(obj.out_file, '\n');
            fprintf(obj.out_file, '%.10f ', obj.transform_state(x, t));
            fprintf(obj.out_file, '\n');
            fprintf(obj.out_file, '%d ', state_ind - 1);
            fprintf(obj.out_file, '\n');
          else %use_net == 2
            obj.write_state(x,t);
            state_ind = round(obj.get_action()) + 1;
          end
          current_target_state = state_ind;

        end

        global max_hip_angle;
        global max_knee_angle;
        global state_targets;

        %targets = state_targets{current_target_state};
        targets = obj.feedback_adjust_state(current_target_state, x);

        % left leg, left knee, left ankle, right leg, right knee, right ankle
        joint_inds = [4;5;6;7;8;9];
        joint_vel_inds = 9 + joint_inds;
        actuals = x(joint_inds);
        vels = x(joint_vel_inds);
        p_const = 200;
        %alphas_p = p_const*[0.1; 1; 1; 1; .1; .1];
        alphas_p = p_const*[1; 1; 0.1; 1; 1; 0.1];
        alphas_d = 2*sqrt(alphas_p); %d_const*[10; 1; 1; 1; 1; 1];
        u = alphas_p .* (targets - actuals) - alphas_d .* vels;

        torso_target = 0.1;
        torso_swing_p = p_const;
        torso_swing_d = 2*sqrt(torso_swing_p);
        torso_im_torque = torso_swing_p * (torso_target - x(3)) - torso_swing_d * x(12);
        if current_target_state == 1 | current_target_state == 2
          stance_ind = 4;
          kick_ind = 1;
        else
          stance_ind = 1;
          kick_ind = 4;
        end

        u(stance_ind) = -torso_im_torque - u(kick_ind);

        % u = u + 3*randn(6,1);
        max_abs_torque = 200;
        u = min(max(u, -max_abs_torque),max_abs_torque);
      end


      %u = Point(obj.p.getInputFrame())
      %[H,C,B] = obj.p.manipulatorDynamics(x(1:2),x(3:4));
      %l = obj.p.l1+obj.p.l2;
      %b = .1;
      %u = C + H*[-obj.p.g*sin(x(1))/l - b*x(3);0];
    end
  end

end
