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
#regard unknown as a particular attribute
def data_preprocessing_attribute(data, features, continuous):
    for var in features:
        if var in continuous:
            media = data[var].median() #replace with median
            data[var] = data[var].apply(lambda x:"no" if x < media else 'yes')

    return data

train_data = pd.read_csv("D:\\EIC-Code\\00-Python\\Machine-Learning-HW\\DecisionTree\\bank\\train.csv",header = None, 
names = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'])

test_data = pd.read_csv("D:\\EIC-Code\\00-Python\\Machine-Learning-HW\\DecisionTree\\bank\\test.csv", header = None, 
names = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'])

features = ['age', 'job', 'marital','education', 'default', 'balance', 'housing','loan', 'contact', 'day','month', 
            'duration','campaign','pdays','previous', 'poutcome']

continuous = ['age', 'balance', 'day','duration','campaign','pdays','previous']

#load data
train_data_att = data_preprocessing_attribute(train_data.copy(), features, continuous)
test_data_att = data_preprocessing_attribute(test_data.copy(), features, continuous)

train_data_att['y'] = train_data_att['y'].map(lambda label: 1 if label == 'yes' else -1) 
test_data_att['y'] = test_data_att['y'].map(lambda label: 1 if label == 'yes' else -1) 

# #-----------------------------Adaboost
if __name__ == "__main__":
    X_train= np.array(train_data_att[features])
    y_train = np.array(train_data_att['y'])
    X_test = np.array(test_data_att[features])
    y_test = np.array(test_data_att['y'])

    features = features

    # Train AdaBoost with decision stumps
    num_iterations = 500 #500

    train_errors, test_errors, stump_train_errors, stump_test_errors = adaboost_stumps(X_train, y_train, features, num_iterations, X_test, y_test)

    # Plot the results
    plot_results(train_errors, test_errors, stump_train_errors, stump_test_errors, num_iterations)

#--------------------------------------------bagging

if __name__ == "__main__":
    X_train = np.array(train_data_att[features])
    y_train = np.array(train_data_att['y'])
    X_test = np.array(test_data_att[features])
    y_test = np.array(test_data_att['y'])

    features = features
    tree_counts = 500 #500 # Number of trees
    max_depth = len(features)  # User-defined max depth (None means fully grown trees, set to a number for limited depth)
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

# # #bias and variance
if __name__ == "__main__":
    # Load the dataset (replace with actual data)
    X_train = np.array(train_data_att[features])
    y_train = np.array(train_data_att['y'])
    X_test = np.array(test_data_att[features])
    y_test = np.array(test_data_att['y'])
    
    features = features
    num_iterations = 100 #100 #100
    num_trees = 500 #500 #500
    num_samples = 1000 #1000
    max_depth = len(features)

    feature_map = {feature: idx for idx, feature in enumerate(features)}
    most_common_label = np.bincount(y_train + 1).argmax() - 1  # Calculate the most common label in the training set
    n_test_examples = X_test.shape[0]

    # Run iterations in parallel
    results = Parallel(n_jobs=-1)(delayed(run_iteration_bagging)(X_train, y_train, X_test, y_test, features, num_trees, max_depth, num_samples, feature_map, most_common_label)
                                  for i in range(1, tree_counts + 1))
    # Collect results
    single_tree_predictions = np.array([res[0] for res in results])
    bagged_tree_predictions = np.array([res[1] for res in results])

    # Compute bias and variance for single tree learner
    single_tree_bias, single_tree_variance = compute_bias_variance(single_tree_predictions, y_test)
    
    # Compute bias and variance for bagged trees
    bagged_tree_bias, bagged_tree_variance = compute_bias_variance(bagged_tree_predictions, y_test)

    # Print results
    print(f"Single Tree Bias: {single_tree_bias}, Variance: {single_tree_variance}")
    print(f"Bagged Trees Bias: {bagged_tree_bias}, Variance: {bagged_tree_variance}")

    # Calculate and print the general squared error
    single_tree_error = single_tree_bias + single_tree_variance # squared error = bias + variance
    bagged_tree_error = bagged_tree_bias + bagged_tree_variance

    print(f"Single Tree General Squared Error: {single_tree_error}")
    print(f"Bagged Tree General Squared Error: {bagged_tree_error}")

    plot_bagging(single_tree_bias, single_tree_variance, bagged_tree_bias, bagged_tree_variance)


# #-------------------------------------------------------------random forest
if __name__ == "__main__":
    X_train = np.array(train_data_att[features])
    y_train = np.array(train_data_att['y'])
    X_test = np.array(test_data_att[features])
    y_test = np.array(test_data_att['y'])

    features = features
    tree_counts = 500 #500 #500  # Number of trees
    feature_subsets = [2,4,6]#[2, 4, 6]  # Feature subset sizes
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

if __name__ == "__main__":
    X_train = np.array(train_data_att[features])
    y_train = np.array(train_data_att['y'])
    X_test = np.array(test_data_att[features])
    y_test = np.array(test_data_att['y'])

    features = features
    num_iterations = 100 #100 #100
    num_trees = 500 #500
    num_samples = 1000
    feature_subsets = [2, 4, 6]  # Feature subset sizes #2, 
    max_depth = len(features)

    feature_map = {feature: idx for idx, feature in enumerate(features)}
    most_common_label = np.bincount(y_train + 1).argmax() - 1  # Calculate the most common label in the training set
    n_test_examples = X_test.shape[0]

    single_tree_bias_variance_squared_error = []
    bagged_tree_bias_variance_squared_error = []


    # Loop over feature subsets in parallel
    for num_features in feature_subsets:
        results = Parallel(n_jobs=-1)(delayed(run_iteration_rf)(
            X_train, y_train, X_test, y_test, features, num_trees, max_depth, num_samples, num_features, feature_map, most_common_label
        ) for _ in range(num_iterations))
       

        # Collect predictions from all iterations
        single_tree_predictions = np.array([res[0] for res in results])
        bagged_tree_predictions = np.array([res[1] for res in results])

        # Compute bias and variance for single tree learner
        single_tree_bias, single_tree_variance = compute_bias_variance(single_tree_predictions, y_test)
        
        # Compute bias and variance for bagged trees
        bagged_tree_bias, bagged_tree_variance = compute_bias_variance(bagged_tree_predictions, y_test)

        print(f"Features: {num_features}, Single Tree Bias: {single_tree_bias}, Variance: {single_tree_variance}, Bagged Trees Bias: {bagged_tree_bias}, Variance: {bagged_tree_variance}")
        
        # Calculate and print the general squared error
        single_tree_error = single_tree_bias + single_tree_variance # squared error = bias + variance
        bagged_tree_error = bagged_tree_bias + bagged_tree_variance

        single_tree_bias_variance_squared_error.append([single_tree_bias, single_tree_variance, single_tree_error])
        bagged_tree_bias_variance_squared_error.append([bagged_tree_bias, bagged_tree_variance, bagged_tree_error])

        print(f"Single Tree General Squared Error: {single_tree_error}")
        print(f"Bagged Tree General Squared Error: {bagged_tree_error}")

    plot_rf_bias_variance (single_tree_bias_variance_squared_error, bagged_tree_bias_variance_squared_error)