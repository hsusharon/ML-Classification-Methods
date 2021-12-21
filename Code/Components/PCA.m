function PCA_error_rate = PCA(data)
    %%we first forget about the labels of the training data
    %%compute the mean of every dimension
    %%compute the covariance matrix of the whole dataset
    %%take the eigenvalues and eigenvectors of the covariance matrix
    %%project the data onto the the new subspace
    PCA_error_rate = 0;

    sum = 0;
    index = 1;
   
    data_set = [data(1).training data(1).testing];
    
    for i = 2:200   %%we first construct a matrix with the entire dataset
        A = data(i).training;
        B = data(i).testing;
        data_set = [data_set A B];
    end
    mean_data = mean(data_set, 'all');
    data_set  = data_set - mean_data;
    %dataset_mean = mean(data_set, 2); %% find the mean of the dataset
    dataset_cov = cov(data_set.');  %% find the covariance of the matrix
    
    %find the eigen value and eigen vector 
    %then extract the 295 vector that has positive eigen values
    [V, D] = eig(dataset_cov);
    point = 1;
    for i = 400:-1:106
        proj_matrix(:,point) = V(:, i);
        point = point+1;
    end
    
    %%project the dataset
    proj_dataset = proj_matrix.' * data_set; %% project the data
    
    %%saperate the dataset of the combined dataset
    index = 1;  
    reduc_dim_data = struct('training', 0, 'testing', 0);
    for i = 1:600
        if mod(i, 3) == 0
            reduc_dim_data(index).testing = proj_dataset(:,i);
            index = index+1;
        elseif mod(i, 3) == 1
            reduc_dim_data(index).training = proj_dataset(:,i);
        else
            reduc_dim_data(index).training = [reduc_dim_data(index).training proj_dataset(:,i)];
        end
    end
    
    %%plug into ML estimation and apply bayes classifier
    PCA_ML_output = ML_estimation(reduc_dim_data, 295);
    PCA_error_rate = Bayes_classifier(PCA_ML_output, reduc_dim_data, 295, 2);

end