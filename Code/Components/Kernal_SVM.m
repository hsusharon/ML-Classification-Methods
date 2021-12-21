function SVM_error_rate = Kernal_SVM(data_struct, mode)
    sigma = 0.00001;
    gamma = 20;
    C = 2;
    %combine the two data with labels
    data_training_set = [data_struct(1).training data_struct(2).training];
    data_training_set = data_training_set';
    for i = 1:300
        if i <151
            label_y(i,1) = 1;
        else
            label_y(i,1) = -1;
        end
    end
    
    test = zeros(length(data_training_set),1);
    gscatter(data_training_set, test,label_y, 'bg');
    %find the optimal mu
    %then we can calculate the optimal theta
    %then we can solve theta0 with theta
    %mu = zeros(1,300);
  
    %find the Kernel K
    K = Kernel_construct(data_training_set,sigma, gamma, mode);

    %find the maximum value of mu (quadradic progamming)
    for i = 1:300  %% construct the H matrix
        for j = 1:300
            H(i,j) = label_y(i) * label_y(j) * K(i,j);
        end
    end
    f = -ones(300,1); %%construct f 
    Aeq = label_y.';
    beq = 0;
    lb = zeros(300,1);
    ub = C*ones(300,1);

    %%quadradic programming
    mu_opt = quadprog(H,f,[],[],Aeq, beq, lb, ub);
    
    data_testing = [data_struct(1).testing data_struct(2).testing];
    data_testing = data_testing';

    for i = 1:100  %%set the label of the testing dataset
        if i <51
            label_y_testing(i,1) = 1;
        else
            label_y_testing(i,1) = -1;
        end
    end
  

    %%prediction for the testing dataset
    error_rate=0;
    if mode == 1  %%RBF case
        for i = 1:100
            sum = 0;
            for j = 1:300
                sum = sum + mu_opt(j) * label_y(j) * exp((-1/sigma^2) * (data_testing(i)-data_training_set(j))^2);
            end
            if label_y_testing(i) * sum <0
                error_rate = error_rate + 1;
            end
        end
    else   %%polynomial case
        for i = 1:100
            sum = 0;
            for j = 1:300
                sum = sum + mu_opt(j) * label_y(j) * (data_testing(i)*data_training_set(j)+1)^gamma;
            end
            if label_y_testing(i) * sum <0
                error_rate = error_rate + 1;
            end
        end
    end
    
    SVM_error_rate = error_rate;
    
end