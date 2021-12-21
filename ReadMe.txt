The objection of the project is to classify the given dataset with different classification methods without any toolbox.
(Detailed information and results are in the report).

Run it with matlab (No machine learning toolbox needed)
Require Optimization Toolbox

For main function
When you run main.m, it willrun all 6 different classifiers

KNN classifiers(line 37): K_pool is equivalent to K in KNN  K>=1
To adjust the value to get different numbers of polls from the testing data

Kernal SVM (line 74) : mode=1,2
mode=1 is to use RBF kernel mode=2 is to use polynomial kernel
To adjust sigma or gamma, go to file ../Components/Kernal_SVM.m for line 2&3
