% this is used to find the quadratic expansion of hand to ball distance squared

function A = estimate_hessian(obj,x)
  eps = 1e-20;
  A = zeros(10,10);
  for i = 1:10
    for j = 1:10
      di = zeros(10,1);
      di(i) = eps;
      dj = zeros(10,1);
      dj(j) = eps;
      A(i,j) = final_state_con(x + di + dj) - final_state_con(x + di) - final_state_con(x + dj) + final_state_con(x);
    end
  end
  A = A/eps^2
end


function [f,df] = final_state_con(obj,x)
  q = x(1:5);
  qd = x(6:10);
  kinsol = obj.doKinematics(q);

  % body index, so p.body(3) is the lower link
  hand_body = 3;

  % position of the "hand" on the lower link, 2.1m is the length
  pos_on_hand_body = [0;-2.1];

  % Calculate position of the hand in world coordinates
  % the gradient, dHand_pos, is the derivative w.r.t. q
  [hand_pos,dHand_pos] = obj.forwardKin(kinsol,hand_body,pos_on_hand_body);

  d2 = norm(hand_pos - x(3:4))^2

  % % ********YOUR CODE HERE ********
  % % Calculate f and the gradient df/dx
  % % f should be [0;0] if and only if the hand_pos calculated above equals
  % % the current position of the ball
  % % DO NOT simply pre-calculate the position of the ball at t=3
  % % the final time of the trajectory might not be 3!

  %f = [hand_pos(1) - x(3); hand_pos(2) - x(4)]
  % %df = [dH_dt(1) - x(8); dH_dt(2) - x(9)]
  f = hand_pos - x(3:4);
  A = zeros([2,10]);
  A(1, 3) = 1;
  A(2, 4) = 1;
  df = [dHand_pos zeros([2 5])]  - A;
  %x, f, df

  % *******************************
end
