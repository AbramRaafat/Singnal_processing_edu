T = 1;         
num_steps = 200; 
initial_pos = [40; 140; 50]; 
speed = 2;       

% State transition matrix 
F = [1 T 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 T 0 0;
     0 0 0 1 0 0;
     0 0 0 0 1 T;
     0 0 0 0 0 1];

% Measurement matrix 
C = [1 0 0 0 0 0;
     0 0 1 0 0 0;
     0 0 0 0 1 0];


Q = diag([0, 0.5, 0, 0.5, 0, 0.5]);
R = randi(3);


[true_states, measurements] = generate_random_path(initial_pos, speed, num_steps, T, R);


x0 = [true_states(1:2:6,1); true_states(2:2:6,1)];
P0 = 1e4 * eye(6);

% Kalman filter
[estimates, ~] = kalman_filter(F, C, Q, R, measurements, x0, P0);

true_pos = true_states([1,3,5], :)';
meas_pos = measurements';
est_pos = estimates([1,3,5], :)';

figure;
hold on; grid on;


plot3(true_pos(:,1), true_pos(:,2), true_pos(:,3), ...
    'g-', 'LineWidth', 2, 'DisplayName', 'True Path');

scatter3(meas_pos(:,1), meas_pos(:,2), meas_pos(:,3), ...
    'r', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Measurements');


plot3(est_pos(:,1), est_pos(:,2), est_pos(:,3), ...
    'b--', 'LineWidth', 1.5, 'DisplayName', 'Kalman Estimate');

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Random Path Tracking with Kalman Filter');
legend; view(3); axis equal;

function [true_states, measurements] = generate_random_path(initial_pos, speed, num_steps, T, R)
    true_states = zeros(6, num_steps);
    measurements = zeros(3, num_steps);
    
    vel = randn(3,1); 
    vel = vel/norm(vel) * speed;  
   
    pos = initial_pos;
    for t = 1:num_steps
        true_states(:,t) = [pos(1); vel(1); pos(2); vel(2); pos(3); vel(3)];
        measurements(:,t) = pos + sqrt(R)*randn(3,1);
        pos = pos + vel*T;

        vel = vel + 0.8*randn(3,1);  
        vel = vel/norm(vel) * speed; 
    end
end




