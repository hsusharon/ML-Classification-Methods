function MDA_error_rate = MDA(data_struct, ML_output)
    
    %%the goal is to find the eigen value of the between class scatter
    %%matrix and the within class matrix
    %%we have to first find the anchor point and calculate the matrixes
    %%after finding the eigen vect8ors, we can project to data and find the
    %%best find gaussian through it
    %%project to testing data and find the maximum value of prob.
    
    %%since mu_i is the same as ML_output.mean
    
    prior = 1/200; %%p(wi) are all equal = 1/200
    num_classes = 200;
    
    anchor = 0;
    for i = 1:200   %%calculate the anchor point mu_0
        anchor = anchor + ML_output(i).mean * prior; 
    end
    
    between_sig=0;
    for i = 1:200  %% calculate the between scatter matrix 504*504
        between_sig = between_sig + prior * (ML_output(i).mean - anchor) * (ML_output(i).mean - anchor).';
    end
    
    within_sig = 0;
    for i = 1:200  %%calculate the within scatter matrix 504*504
        sigma_i = 0;
        for j = 1:2
            sigma_i = sigma_i + prior * (data_struct(i).training(:,j) - ML_output(i).mean) * (data_struct(i).training(:,j) - ML_output(i).mean).';
        end
        within_sig = within_sig + sigma_i/2;
    end
    
    within_sig  = within_sig + 0.3*eye(504);  %%normalize the within scatter matrix to make it invertable


    [V, D] = eig(between_sig, within_sig);
    
    for i = 1:num_classes+2  %%extract the first 199 vector that has the maximum eigenvalue
        A(:, i) = V(:,i);
    end
    
    
    MDA_project_data = struct('training', 0, 'testing', 0);

    for i = 1:num_classes  %%project the data onto the eigenvectors
        MDA_project_data(i).training = A.' * data_struct(i).training;
        MDA_project_data(i).testing = A.' * data_struct(i). testing;
    end
    
    %%pass into Bayes classifier
    MDA_ML_output = ML_estimation(MDA_project_data, num_classes+2);
    MDA_error_rate = Bayes_classifier(MDA_ML_output, MDA_project_data, num_classes+2, 1);

end