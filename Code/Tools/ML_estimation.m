function [ML_output] = ML_estimation(data_struct, data_dimen)
    %%We know that in class, the ML estimation of a guassian 
    %%the mean is the average of the training data set 
    %%the variance is the training dataset-mean
    total = 0;
    for i = 1:200
        total = sum(data_struct(i).training, 2);
        ML_output(i) = struct('mean', total/2, 'cov', 0);
    end
   
    %%workspace;
    %%find the covariance matrix
    inden = 0.3*eye(data_dimen);
    for i = 1:200
        covariance = 0;
        for j = 1:2
            covariance = covariance + (data_struct(i).training(:,j) - ML_output(i).mean) * (data_struct(i).training(:,j) - ML_output(i).mean).';
        end
            ML_output(i).cov = (covariance/2)+inden;
    end
    
end