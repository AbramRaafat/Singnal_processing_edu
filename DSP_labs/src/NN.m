function NN
%NETBP Uses backpropagation to train a network

%%%%%%% DATA %%%%%%%%%%%
noise_level = 0.1;          
num_samples = 1000;          
rng(5000);                   

X_ideal = [ 0 0 0;  0 0 1;  0 1 0;  0 1 1; 1 0 0;  1 0 1;  1 1 0;  1 1 1]';  
Y_ideal = [0; 1; 1; 0; 1; 0; 0; 1];  

X = zeros(3,num_samples);        
y = zeros(2,num_samples);         
for i = 1:num_samples
    idx      = randi(8);                      
    noisy    = X_ideal(:,idx) + (2*rand(3,1)-1)*noise_level;
    X(:,i)   = min(max(noisy,0),1);            
    target   = Y_ideal(idx);                   
    y(:,i)   = [target==0 ; target==1];         
end
x1 = X(1,:);    x2 = X(2,:);    x3 = X(3,:);

% Initialize weights and biases
W2 = 0.5*randn(2,3);   W3 = 0.5*randn(3,2);   W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1);   b3 = 0.5*randn(3,1);   b4 = 0.5*randn(2,1);

% Forward and Back propagate
eta   = 0.05;                        % learning rate
Niter = 5e5;                         % number of SG iterations
savecost = zeros(Niter,1);           % value of cost function at each iteration
print_interval = 1e4;                
for counter = 1:Niter
    k = randi(num_samples);          %  choose a training point at random
    x = [x1(k); x2(k); x3(k)];
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4 - y(:,k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    newcost = cost_vec(W2,W3,W4,b2,b3,b4);          % display cost to screen
    savecost(counter) = newcost;
    if mod(counter,print_interval)==0
        fprintf('Iter %d | cost %.4f\n',counter,newcost);
    end
end


fprintf('\nPredictions:\n');
X_test = zeros(3, 8);
for i = 1:8
    noisy = X_ideal(:,i) + (2*rand(3,1)-1)*noise_level;
    X_test(:,i) = min(max(noisy,0),1);
end
a2 = activate(X_test, W2, b2);
a3 = activate(a2, W3, b3);
a4 = activate(a3, W4, b4);
for i = 1:8
    pred = a4(2,i); % Probability of class 1
    rounded = round(pred);
    fprintf('Input: [%.2f %.2f %.2f], Predicted : %.4f, Rounded: %d, Actual: %d\n', ...
            X_test(1,i), X_test(2,i), X_test(3,i), pred, rounded, Y_ideal(i));
end
accuracy = mean(round(a4(2,:)) == Y_ideal') * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

% Show decay of cost function
figure, semilogy(1:print_interval:Niter,savecost(1:print_interval:Niter))
xlabel('iteration'), ylabel('MSE cost')
title('XOR logic gate training ')

function costval = cost_vec(W2,W3,W4,b2,b3,b4)
    a2 = activate(X,  W2,b2);        
    a3 = activate(a2, W3,b3);        
    a4 = activate(a3, W4,b4);       
    costval  = norm(y - a4,'fro')^2;  
end

function y = activate(x,W,b)
%ACTIVATE Evaluates sigmoid function.
%
% x is the input vector, y is the output vector
% W contains the weights, b contains the shifts
%
% The ith component of y is activate((Wx+b)_i)
% where activate(z) = 1/(1+exp(-z))
y = 1./(1+exp(-(W*x+b)));
end
end