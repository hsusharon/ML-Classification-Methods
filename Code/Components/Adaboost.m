function Adaboost_error_rate = Adaboost(data_struct)
    total_training_samples = 300;

    %%classify data
    data_training_set = [data_struct(1).training data_struct(2).training]; %% 300*1 training dataset
    data_training_set = data_training_set';
    for i = 1:300  %%setting the known labels
        if i <151
            label_y(i,1) = 1;
        else
            label_y(i,1) = -1;
        end
    end
    
    for i = 1:total_training_samples  %initialize weight
        W(i) = 1/total_training_samples;
    end
    error_counted = 0;
    %%adaboost algorithm
    iteration = 1;
    while iteration <200  
        sum_weight = sum(W);
        P = W / sum_weight; %%normalize weight

        %find theta such that < 1/2 and save the theta(mu)
        selected_data_sample = randsample(length(W), (2/3)*length(W), true, P);
        selected_data_sample = sort(selected_data_sample);

        for i=1:length(selected_data_sample)  %%collect the data that are selected to construct the Linear SVM
            data_selected_training(iteration, i) = data_training_set(selected_data_sample(i));
            data_label_y(iteration,i) = label_y(selected_data_sample(i));
        end
        
        mu(iteration,:) = linear_SVM_adaboost(data_selected_training(iteration,:), data_label_y(iteration,:));

        error=0;
        for i = 1:total_training_samples  %%calculation error
            total = 0;
            for j = 1:length(selected_data_sample)
                total = total + mu(iteration,j) * data_label_y(iteration,j) * data_training_set(i) * data_selected_training(iteration,j);
            end
            phi(i) = total;
            if label_y(i) ~= sign(total)
                error = error + P(i);
            end
        end

        if error<0.5

            a(iteration) = (1/2) * log((1-error)/error);  %%find a for every iteration
        
        
            for i=1:total_training_samples
                W(i) = W(i) * exp(-1 * label_y(i) * a(iteration) * phi(i));
            end

            iteration  = iteration +1;
        end
    end
    

    %prediction
   data_testing = [data_struct(1).testing data_struct(2).testing]; % 1*100 matrix
   for i = 1:100  %the label for training datasets
       if i<51
           y_label_testing(i) = 1;
       else 
           y_label_testing(i) = -1;
       end
   end

   Adaboost_error_rate = 0;
   
   for i = 1:length(y_label_testing)  %% run the testing data and calculate the error rate
       total = 0;
       for j = 1:iteration-1
           theta = 0;
           for k = 1:200
                theta = theta + data_testing(i) * mu(j,k) * data_selected_training(j,k) * data_label_y(j,k);
           end
           total = total + a(j) * theta;
       end
       record(i) = sign(total);
       if sign(total) ~= y_label_testing(i)
           Adaboost_error_rate = Adaboost_error_rate + 1;
       end
   end


    
end