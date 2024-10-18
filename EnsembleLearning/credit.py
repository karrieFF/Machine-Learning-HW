import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
np.random.seed(42)

from Adaboost_bank import adaboost_stumps, plot_results
from Randomforest_bank import run_iteration_rf, compute_bias_variance, plot_rf_bias_variance
from Bagging_bank import bagging_stamp, run_iteration_bagging,compute_bias_variance, plot_bagging, plot_bagging_tree2
from Randomforest_bank import train_single_tree, calculate_errors

#-----------------------------------------------load data
#load data
data = pd.read_excel("D:\\EIC-Code\\00-Python\\Machine-Learning-HW\\EnsembleLearning\\default of credit card clients.xls")
data.drop(columns = 'ID', inplace = True) #drop ID

#check continuous variable
threshold = 11
continuous_column = [col for col in data.columns if data[col].nunique() > threshold]

#replace continuous variable
for col2 in continuous_column:
    media = data[col2].median() #replace with median
    data[col2] = data[col2].apply(lambda x:0 if x < media else 1)

#Split data
n_train = 24000
n_test = 6000
n_samples = data.shape[0]
indices = np.random.choice(n_samples, size=n_train, replace=False)
out_of_sample_indices = np.setdiff1d(np.arange(n_samples), indices)

train_data = np.array(data.iloc[indices])
test_data = np.array(data.iloc[out_of_sample_indices])

X_train = train_data[:,0:-1]
y_train = train_data[:,-1]
y_train[y_train == 0] = -1 #replace 0 to 1 to keep consistent
X_test = test_data[:,0:-1]
y_test = test_data[:,-1]
y_test[y_test == 0] = -1 #replace 0 to 1 to keep consistent
features = data.columns[0:-1]

#-------------------------------------Adaboost
if __name__ == "__main__":
    # Train AdaBoost with decision stumps
    num_iterations = 100 #500

    train_errors, test_errors, stump_train_errors, stump_test_errors = adaboost_stumps(X_train, y_train, features, num_iterations, X_test, y_test)

    # Plot the results
    plot_results(train_errors, test_errors, stump_train_errors, stump_test_errors, num_iterations)

#--------------------------------------------bagging

if __name__ == "__main__":
    tree_counts = 100 #500 # Number of trees
    max_depth = len(features) # User-defined max depth (None means fully grown trees, set to a number for limited depth)
    feature_map = {feature: idx for idx, feature in enumerate(features)}
    most_common_label = np.bincount(y_train + 1).argmax() - 1  # Calculate the most common label in the training set
    
    train_errors, test_errors = bagging_stamp(X_train,y_train,
                                              features,
                                              feature_map, 
                                              max_depth, 
                                              tree_counts, 
                                              X_test, 
                                              y_test, 
                                              most_common_label)
    
    # Final cumulative errors after all trees are added
    print(f"Final Train Errors: {train_errors[-1]}")
    print(f"Final Test Errors: {test_errors[-1]}")
    plot_bagging_tree2(tree_counts,train_errors,test_errors)

#-------------------------------------------------------------random forest
if __name__ == "__main__":
    tree_counts = 100 #500 #500  # Number of trees
    feature_subsets = [2,4,6] #[2, 4, 6]  # Feature subset sizes
    max_depth = len(features) 

    # Mapping features to their indices
    feature_map = {feature: idx for idx, feature in enumerate(features)}
    most_common_label = np.bincount(y_test + 1).argmax() - 1  # Shift back
    

    for num_features in feature_subsets:
        train_errors = []
        test_errors = []
        forest = []
        
        # Train all 500 trees in parallel
        trees = Parallel(n_jobs=-1)(delayed(train_single_tree)(i, X_train, y_train, features, feature_map, max_depth, num_features)
                                    for i in range(tree_counts))
        n_samples = X_train.shape[0]
        
        # Sequentially add trees to the forest and calculate cumulative errors
        for i in range(tree_counts):
            single_tree, indice = trees[i]
            out_of_sample_indices = np.setdiff1d(np.arange(n_samples), indice)
            forest.append(single_tree)  # Add the trained tree to the forest
            
            # Calculate errors with the current forest
            train_error, test_error = calculate_errors(forest, X_train[out_of_sample_indices], y_train[out_of_sample_indices], X_test, y_test, feature_map, most_common_label)
            
            # Store errors
            train_errors.append(train_error)
            test_errors.append(test_error)

            # Print errors at each step
            #print(f"Trees: {i+1}, Train Error: {train_error}, Test Error: {test_error}")

        print(f"num_features:{num_features}, Train Error: {train_errors}, Test Error: {test_errors}")
        # Plot the errors for the current feature subset size
        plt.plot([i+1 for i in range(len(train_errors))], train_errors, label=f'Train Error (features={num_features})')
        plt.plot([i+1 for i in range(len(test_errors))], test_errors, label=f'Test Error (features={num_features})')

    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title('Training and Test Errors vs Number of Trees')
    plt.legend()
    plt.show()
