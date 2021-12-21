function KNN_data_error = KNN_Rule(K, data_struct)
    %%we can poll among the k nearest neighbors of x
    %%first put training data into a 1d array
    %%then we calculate the distance of the testing data then take the poll of it
    %%since each class has only two training data, so only K=1 is suitable
    %%in this case
    KNN_data_error = 0;

    for i = 1:200
        for j = 1:200
            distance_array(2*(j-1)+1) = Eucli_dist(data_struct(i).testing, data_struct(j).training(:,1));
            distance_array(2*j) = Eucli_dist(data_struct(i).testing, data_struct(j).training(:,2));
        end
        [B, I] = sort(distance_array);
        I = floor((I-1)/2)+1;
        class(i) = mode(I(1:K));
        if class(i) ~= i
            KNN_data_error = KNN_data_error + 1;
        end
    end
    
    KNN_data_error = KNN_data_error / 200;

end