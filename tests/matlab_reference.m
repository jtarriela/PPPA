% MATLAB reference test for PPPA verification
% Generates spherical test instance and computes HV with mexPPPA
% Saves data and results for Python comparison

clear; clc;

% Fixed seed for reproducibility
rng(42);

pop = 100;
dim = 13;

bounds = ones(1, dim);

% Generate points on unit sphere (spherical instance)
M = 0;
C = zeros(pop, dim);
for i = 1:dim-1
    C(:,i) = sqrt(1 - M) .* rand(pop, 1);
    M = M + C(:,i).^2;
end
C(:,dim) = sqrt(1 - M);

% Flip points (minimization format)
C = repmat(bounds, pop, 1) - C;

% Save the test data for Python to load
save('test_data.mat', 'C', 'bounds', 'pop', 'dim');

% Compute HV using mexPPPA
fprintf('Computing hypervolume with mexPPPA...\n');
tic;
hv_result = mexPPPA(C, bounds);
elapsed_time = toc;

fprintf('Hypervolume: %.10f\n', hv_result);
fprintf('Time elapsed: %.4f seconds\n', elapsed_time);

% Save results
save('test_results.mat', 'hv_result', 'elapsed_time', 'C', 'bounds');
fprintf('Results saved to test_results.mat\n');
