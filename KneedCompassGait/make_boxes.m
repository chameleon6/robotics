function [boxes, h] = make_boxes(xs)
  h = 0.05;
  n = size(xs,1);
  boxes = zeros(0,5);
  for i = 1:(n-1)
    x_center = (xs(i) + xs(i+1)) / 2;
    width = xs(i+1) - xs(i);
    new_box = [x_center 0 width 100 (n-i)*h];
    boxes = [boxes; new_box];
  end

end
