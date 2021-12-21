function Kernel_matrix = Kernel_construct(data, sigma, gamma, mode)
    
    if mode == 1  %% mode=1 exp kernel

        for i = 1:length(data)
            for j = 1:length(data)
                Kernel_matrix(i,j) = exp((-1/sigma^2)* (data(i)-data(j))^2);
            end
        end

    elseif mode == 2 %%mode=2 polynomial kernel
        for i = 1:length(data)
            for j = 1:length(data)
                Kernel_matrix(i,j) = (data(i)*data(j)+1)^gamma;
            end
        end
    else   %%mode=3 linear SVM
        for i = 1:length(data)
            for j = 1:length(data)
                Kernel_matrix(i,j) = data(i)*data(j);
            end
        end
    end 

end
