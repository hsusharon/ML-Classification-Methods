function distance = Eucli_dist(data1, data2)

    differ = data1 - data2;
    distance = sqrt(differ.' *differ);
    
end