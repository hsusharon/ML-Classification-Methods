addpath('./Tools');
addpath('./Components');
addpath('./Data');

%% Load provided dataset
load('data.mat');
facial_index = 200;  %%the number of different faces

for n = 1:facial_index
    data_struct(n) = struct('neutral', face(:,:,3*n-2), 'express', face(:,:,3*n-1), 'illum', face(:,:,3*n));
end

%%reshaping the data and seperate the testing and training data
for i = 1:200
    reshape_training_struct1 = reshape(data_struct(i).neutral, [504,1]);
    reshape_training_struct2 = reshape(data_struct(i).express, [504,1]);
    reshape_training_struct = [reshape_training_struct1 reshape_training_struct2];
    reshape_testing_struct = reshape(data_struct(i).illum, [504,1]);
    data_reshape(i) = struct('training', reshape_training_struct(:,:,:), 'testing', reshape_testing_struct(:,:,:));
end

%% Bayes' Classifier
%%find the ML estimation of every training data
ML_output = ML_estimation(data_reshape, 504);

%%find the baysian classifier
Bayes_error_rate = Bayes_classifier(ML_output, data_reshape, 504, 2);
Bayes_error_rate


%% MDA method with baysian classifier
MDA_error_rate = MDA(data_reshape, ML_output);
MDA_error_rate


%% KNN classifier
K_poll = 1;
KNN_error_rate = KNN_Rule(K_poll, data_reshape);
KNN_error_rate


%% PCA mathod with baysian classifier
PCA_error_rate = PCA(data_reshape);
PCA_error_rate


%% Reconstruct data for 2 classification case
reshape_training_neutral = reshape(data_struct(1).neutral, [504,1]);
reshape_training_express = reshape(data_struct(1).express, [504,1]);
reshape_training_illu = reshape(data_struct(1).illum, [504,1]);

for i = 2:150
    reshape_training_neutral = [reshape_training_neutral reshape(data_struct(i).neutral, [504,1])];
    reshape_training_express = [reshape_training_express reshape(data_struct(i).express, [504,1])];
    reshape_training_illu = [reshape_training_illu reshape(data_struct(i).illum, [504,1])];
end

reshape_testing_neutral = reshape(data_struct(151).neutral, [504,1]);
reshape_testing_express = reshape(data_struct(151).express, [504,1]);
reshape_testing_illu = reshape(data_struct(151).illum, [504,1]);
for i = 152:200
    reshape_testing_neutral = [reshape_testing_neutral reshape(data_struct(i).neutral, [504,1])];
    reshape_testing_express = [reshape_testing_express reshape(data_struct(i).express, [504,1])];
    reshape_testing_illu = [reshape_testing_illu reshape(data_struct(i).illum, [504,1])];
end

facial_data_reshape(1) = struct('training', reshape_training_neutral, 'testing', reshape_testing_neutral);
facial_data_reshape(2) = struct('training', reshape_training_express, 'testing', reshape_testing_express);
facial_data_reshape(3) = struct('training', reshape_training_illu, 'testing', reshape_testing_illu);

%%Dimationality reduction (1D)
MDA_SVM_1dim = MDA_SVM_Dim_reduction(facial_data_reshape);

%% Kernal SVM
mode = 1;  %mode=1(RBF)  mode=2(Polynomial)
SVM_error_rate = Kernal_SVM(MDA_SVM_1dim, mode);
SVM_error_rate

%% Adaboost
Adaboost_error_rate =  Adaboost(MDA_SVM_1dim);
Adaboost_error_rate



