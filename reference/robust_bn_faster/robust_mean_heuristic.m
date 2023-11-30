% Robust mean estimation via Filtering:
%   I. Diakonikolas, G. Kamath, D. Kane, J. Li, A. Moitra, A. Stewart.
%   Robust Estimators in High Dimensions without the Computational Intractability.
%   In Proceedings of the 57th IEEE Symposium on Foundations of Computer Science (FOCS), 2016.

% We use only one iteration, which is fast and gives reasonable results.
% Input: X (N x d, N d-dimensinoal samples) and eps (fraction of corruption).
% Output: a hypothesis vector mu (a column vector).


function [mu] = robust_mean_heuristic(X, eps)
% N = number of samples, d = dimension.
N = size(X, 1);
d = size(X, 2);

% A very fast heuristic for robust mean estimation.

% Compute v1 = the top eigenvector of cov(X).
% Project all samples along the direction of v1, and remove eps-fraction of the sample farthest from the projected mean.
covX = cov(X);
[v1, ~] = eigs(covX, 1);

projected_mean = mean(X) * v1;
projection_data_pair = [abs(X * v1 - projected_mean) X];
% Sort by the absolute value of the projection (first column).
sorted_pair = sortrows(projection_data_pair);
% Remove eps-fraction of the sample farthest from the projected mean.
N_minus_epsN = round((1 - eps) * N);
mu = mean(sorted_pair(1:N_minus_epsN, 2:end))';
end