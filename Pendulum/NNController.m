classdef NNController < DrakeSystem

  properties
    p
    matlab_state_file
    python_action_file
  end

  methods
    function obj = NNController(plant)

      obj = obj@DrakeSystem(0,0,2,1,true,false); %%true?
      obj.p = plant;
      obj.matlab_state_file = strcat(pwd,'/matlab_state_file.out');
      obj.python_action_file = strcat(pwd,'/python_action_file.out');
      obj = obj.setInputFrame(plant.getStateFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
    end

    % function x0 = getInitialState(obj)
    %   x0 = [2; 0];
    % end

    % function ts = getSampleTime(obj)
    %   ts = [0.01; 0];
    % end

    function r = reward(obj,x)
      %r = -100*((x(1) - pi)^2 + x(2)^2)

      if cos(x(1)) < -0.9 & abs(x(2)) < 0.1
        r = 100
        fprintf('good x:%f\n', x);
      else
        r = 0
      end

      %r = -100*x(2)^2;
    end

    function write_state(obj,x,t)
      f = fopen(obj.matlab_state_file, 'w');
      x_new = x;
      x_new(1) = mod(x(1), 2*pi);
      fprintf(f, '%d\n', obj.reward(x));
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
        a = fscanf(f, '%f\n');
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

      t, x
      obj.write_state(x,t);
      u = obj.get_action()

    end
  end

end
