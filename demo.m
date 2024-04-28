%
% Define a classification problem of 
% size 10,000 and dimension 100
% The matrix and label is generated randomly
clear;
clc;

d  = 100;
N  = 10000;
A  = randn(N,d);
b  = (sign(randn(N,1)-0.5)+1)/2;
N  = size(A,1);

L  = .8*svds(A,1)^2/N;
AT = A';

% Run experiment 1
exp_1;
% Run experiment 2
exp_2;