function MDA_SVM_1dim = MDA_SVM_Dim_reduction(data_struct)
    %%the goal is to find the eigen value of the between class scatter
    %%matrix and the within class matrix
    %%we have to first find the anchor point and calculate the matrixes
    %%after finding the eigen vectors, we can project to data and find the
    %%best find gaussian through it
    %%project to testing data and find the maximum value of prob.
    
    %%since mu_i is the same as ML_output.mean
    
    %value1 = [data_struct(1).training data_struct(1).testing];
    %value2 = [data_struct(2).training data_struct(2).testing];
    value1 = data_struct(1).training;
    value2 = data_struct(2).training;
    data(1) = struct('value', value1, 'mean', 0, 'cov', 0);
    data(2) = struct('value', value2, 'mean', 0, 'cov', 0);
    
    num_classes = 2;
    prior = 1/num_classes; %%p(wi) are all equal = 1/2

    %%We know that in class, the ML estimation of a guassian 
    %%the mean is the average of the training data set 
    %%the variance is the training dataset-mean
   
    for i = 1:num_classes
        total = sum(data(i).value,2);
        data(i).mean = total/150;
    end

    anchor = 0;
    for i = 1:num_classes   %%calculate the anchor point mu_0
        anchor = anchor + data(i).mean * prior; 
    end
    
    between_sig=0;
    for i = 1:num_classes  %% calculate the between scatter matrix 504*504
        between_sig = between_sig + prior * (data(i).mean - anchor) * (data(i).mean - anchor).';
    end
    
    within_sig = 0;
    for i = 1:num_classes  
        sigma_i = 0;
        for j = 1:150
            sigma_i = sigma_i + prior * (data(i).value(:,j) - data(i).mean) * (data(i).value(:,j) - data(i).mean).';
        end
        within_sig = within_sig + sigma_i/150;
    end
    
    within_sig  = within_sig + 0.1*eye(504);  %%normalize the within scatter matrix to make it invertable


    [V, D] = eig(between_sig, within_sig);
    
    A = V(:,504);

    
    
    MDA_SVM_1dim = struct('training', 0, 'testing', 0);

    for i = 1:num_classes
        MDA_SVM_1dim(i).training = A.' * data_struct(i).training;
        MDA_SVM_1dim(i).testing = A.' * data_struct(i). testing;
    end
    
end