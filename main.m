%
% Datasets are available in 
% https://www.causality.inf.ethz.ch/data/CINA.html and 
% https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets
clear;
clc;
for l = 1:5
    switch l
        case 1
            dataset_name = 'a9a';
        case 2
            dataset_name = 'cina';
        case 3
            dataset_name = 'gisette';
        case 4
            dataset_name = 'real-sim';
        case 5
            dataset_name = 'rcv1';
    end

    try
        train_name   = '_train_scale.mat';
        load(join([dataset_name,train_name]));
    catch
        train_name   = '_train_data.mat';
        load(join([dataset_name,train_name]));
    end
    
    try
        label_name   = '_labels.mat';
        load(join([dataset_name,label_name]));
    catch
        label_name   = '_train_labels.mat';
        load(join([dataset_name,label_name]));
    end
    
    b(b==0)          = -1;
    N                = size(A,1);
    L                = .8*svds(A,1)^2/N;
    AT               = A';

    % Run experiment 1
    exp_1;
    % Run experiment 2
    exp_2;
end
