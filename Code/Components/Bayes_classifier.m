function error_rate = Bayes_classifier(ML_output, data_struct, data_dim, mode)
    %%bayes classifier is to find the maximum p(testing data; ML_pu, ML_cov)
    %%by calculating using the multidimension formula, we can calculate
    %%every probability of the classes 

    exp_const = 1/((2*pi)^(data_dim/2));
    error_rate = 0;
    if mode == 1 %%if MDA
        for i = 1:200  %%calculate p(testing_x ; mean,cov)
            for j = 1:200
                value = (data_struct(i).testing - ML_output(j).mean).' * 1/(ML_output(j).cov) * (data_struct(i).testing - ML_output(j).mean);
                value = value - 14555;
                prob_gaus(j) = exp_const*(1/sqrt(det(ML_output(j).cov)))*exp((-1/2)*value);
            end

            [M, bayes_class(i)] = max(prob_gaus);
            if bayes_class(i) ~= i
                error_rate = error_rate + 1;
            end
        end
    else   %% if bayes classifier
        for i = 1:200  %%calculate p(testing_x ; mean,cov)
            for j = 1:200;
                value = (data_struct(i).testing - ML_output(j).mean).' * 1/(ML_output(j).cov) * (data_struct(i).testing - ML_output(j).mean);
                prob_gaus(j) = exp_const*(1/sqrt(det(ML_output(j).cov)))*exp((-1/2)*value);
             end

            [M, bayes_class(i)] = max(prob_gaus);
            if bayes_class(i) ~= i
                error_rate = error_rate + 1;
            end
        end

    end

    error_rate = error_rate/200;
  
end