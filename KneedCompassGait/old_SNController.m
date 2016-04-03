classdef SNController < DrakeSystem

  properties
    p
    matlab_state_file
    python_action_file
  end

  methods
    function obj = SNController(plant)

      obj = obj@DrakeSystem(0,0,18,6,true,false); %%true?
      obj.p = plant;
      obj.matlab_state_file = strcat(pwd,'/../NN/matlab_state_file.out');
      obj.python_action_file = strcat(pwd,'/../NN/python_action_file.out');
      obj = obj.setInputFrame(plant.getStateFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
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

    function u = output(obj,t,junk,x)

      % if mod(int16(t*100),5) == 0
      %   t
      %   y
      %   coordinates = obj.p.getStateFrame().getCoordinateNames();
      %   for i=1:length(coordinates)
      %     fprintf(1,'%20s = %f\n',coordinates{i},x(i));
      %   end
      % end
      % if mod(int16(t*100),5) == 0
      %   t
      % end

      %qdd = (u - obj.m*obj.g*obj.lc*sin(q) - obj.b*qd)/obj.I

      %p = obj.p;
      %q = mod(x(1),2*pi);
      %qd = x(2);
      %offset_u = p.m*p.g*p.lc*sin(q) + p.b*qd
      %u=(offset_u - p.I*(qd + sqrt(2)*(q-pi))) + 0.01;

      t, x
      obj.write_state(x,t);
      u = obj.get_action()

    end
  end

end
