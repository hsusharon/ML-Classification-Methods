function error_rate = Gaussian(ML_output, data_struct)
        

    for i = 1:200  %%calculate p(x ; mean,cov)

        for j = 1:200
            prob_gaus(i,j) = (1/((2*pi)^252) * sqrt(det(ML_output(j).cov)))*exp((-1/2)*(data_struct(i).testing - ML_output(j).mean).' * 1/(ML_output(j).cov) * (data_struct(i).testing - ML_output(j).mean));
            disp(det(ML_output(j).cov));
        end
    end
    [M, I] = max(prob_gaus,[], 2, 'linear'); %%find the maximum prob
    
    error = 0;
    for i = 1:200  %%calculate error rate
        if I(i) ~= i
            error = error +1;
        end
    end
    
    error_rate = error/200;
  
end