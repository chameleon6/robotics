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

    function ts = getSampleTime(obj)
      ts = [0.001 0; 0 0];
    end

    function r = reward(obj,x)
      r = -(x(1) - pi)^2;
    end

    function write_state(obj,x)
      f = fopen(obj.matlab_state_file, 'w');
      fprintf(f, '%d\n', obj.reward(x));
      fprintf(f, '%d ', x);
      fprintf(f, '\n');
      fclose(f);

      %debug
      fprintf('writing state\n');
      fprintf('%d\n', obj.reward(x));
      fprintf('%d ', x);
      fprintf('\n');
    end

    function a = get_action(obj)
      start_time = cputime;
      while exist(obj.python_action_file, 'file') ~= 2
        if cputime - start_time > 10
          error('timeout')
        end
        continue;
      end

      f = fopen(obj.python_action_file, 'r');
      a = fscanf(f, '%f\n');
      fclose(f);
      delete(obj.python_action_file);

      %debug
      fprintf('read state:\n');
      fprintf('%d ', a);
      fprintf('\n');

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

      obj.write_state(x);
      u = obj.get_action();

    end
  end

end
