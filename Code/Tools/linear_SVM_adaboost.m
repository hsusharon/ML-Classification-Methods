function mu_opt = linear_SVM_adaboost(data, label_y)

    %K = Kernel_construct(data, sigma, gamma, mode);
    %find the maximum value of mu (quadradic progamming)
    for i = 1:length(label_y)  %% construct the H matrix
        for j = 1:length(label_y)
            H(i,j) = label_y(i) * label_y(j) * data(i)*data(j);
        end
    end
    f = -ones(length(label_y),1); %%construct f 
    Aeq = label_y;
    beq = 0;
    lb = zeros(length(label_y),1);
    ub = 5*ones(length(label_y),1);


    %%quadradic programming
    mu_opt = quadprog(H,f,[],[],Aeq, beq, lb, ub);
end