function RBN_3XOR
clc, rng(5000)


var = 1e-3;     
num_samples = 10000;    
K = 8;
X_ideal = [ 0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1 ]';
Y_ideal = [0;1;1;0;1;0;0;1];

X = zeros(3,num_samples);         
y = zeros(num_samples,1);        

for i = 1:num_samples
    idx = randi(K);
    noise = sqrt(var) * randn(3,1);  
    X(:,i) = X_ideal(:,idx) + noise;
    % X(:,i) = min(max(X(:,i),0),1);  
    y(i) = Y_ideal(idx);
end
     

[~,Mu] = kmeans(X',K);  
dmax = max(pdist(Mu));             
Phi = exp(-K * pdist2(X', Mu).^2 / dmax^2);

w = (Phi.'*Phi)\(Phi.'*y); % solving the same way as LS weights

Xtest = X_ideal' + sqrt(var) * randn(size(X_ideal'));
Xtest = min(max(Xtest,0),1);

Phi_test = exp(-K * pdist2(Xtest, Mu).^2 / dmax^2);
yhat = Phi_test * w;
for i = 1:8
    fprintf('Input: [%.2f %.2f %.2f], Predicted: %.4f, Rounded: %d\n', ...
        Xtest(i,1),Xtest(i,2),Xtest(i,3),yhat(i),round(yhat(i)));
end
end



