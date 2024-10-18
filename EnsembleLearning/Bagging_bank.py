import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
np.random.seed(42)

# Function to calculate entropy for any label type
def entropy(X, y):
    class_labels, class_counts = np.unique(y, return_counts=True)
    probabilities = class_counts / len(y)
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding epsilon for stability
    return entropy_value

# Function to calculate information gain for a randomly selected subset of features
def information_gain_fun(X, y, feature_names, feature_map):
    total_entropy = entropy(X, y)

    best_info_gain = -np.inf
    best_feature = None

    for feature in feature_names:
        feature_idx = feature_map[feature]
        expected_entropy = 0
        feature_values = X[:, feature_idx]  # Access column by index
        unique_values = np.unique(feature_values)

        for value in unique_values:
            sub_mask = feature_values == value
            sub_X, sub_y = X[sub_mask], y[sub_mask]
            sub_entropy = entropy(sub_X, sub_y)
            proportion = len(sub_y) / len(y)
            expected_entropy += proportion * sub_entropy

        information_gain = total_entropy - expected_entropy
        if information_gain > best_info_gain:
            best_info_gain = information_gain
            best_feature = feature

    return best_feature

# Function to predict the label for one sample
def predict_one(tree, x, feature_map, most_common_label):
    if not isinstance(tree, dict):
        return tree

    parent_node = next(iter(tree))  # Get the current feature being split on
    subtree = tree[parent_node]
    feature_idx = feature_map[parent_node]
    node_value = x[feature_idx]

    # Handle the case where we encounter a value the tree hasn't seen before
    if node_value in subtree:
        return predict_one(subtree[node_value], x, feature_map, most_common_label)
    else:
        return most_common_label  # Return most common label if value not found in subtree

# Function to predict labels for a dataset
def predict(tree, X, feature_map, most_common_label,  n_jobs=-1):

    return np.array(Parallel(n_jobs=n_jobs)(delayed(predict_one)(tree, x, feature_map, most_common_label) for x in X))


# Function to predict using the Bagged Trees
def bagged_trees_predict(trees, X, feature_map, most_common_label,n_jobs=-1):
    
    tree_predictions = np.array(Parallel(n_jobs=n_jobs)(delayed(predict)(tree, X, feature_map, most_common_label, n_jobs=1) for tree in trees))
    
    # Perform majority voting (aggregate predictions across trees)
    majority_vote = np.sign(np.sum(tree_predictions, axis=0))
    
    return majority_vote

# ID3 Algorithm with support for a user-defined max depth and feature subset selection
def ID3(X, y, feature_names, feature_map, current_depth, max_depth):
    if len(np.unique(y)) == 1:
        return y[0]
    
    # If there are no more features, or if the max depth is reached, return the most common label
    elif len(feature_names) == 0 or (max_depth is not None and current_depth >= max_depth):
        shifted_y = y + 1  # Shift labels to ensure non-negative values for bincount
        most_common_label = np.bincount(shifted_y).argmax() - 1  # Shift back to original labels
        return most_common_label

    else:
        best_feature = information_gain_fun(X, y, feature_names, feature_map)
        tree = {best_feature: {}}

        best_feature_idx = feature_map[best_feature]

        for value in np.unique(X[:, best_feature_idx]):
            sub_mask = X[:, best_feature_idx] == value
            sub_X, sub_y = X[sub_mask], y[sub_mask]

            if sub_X.shape[0] == 0:
                shifted_y = y + 1
                most_common_label = np.bincount(shifted_y).argmax() - 1
                tree[best_feature][value] = most_common_label
            else:
                new_feature_names = [f for f in feature_names if f != best_feature]
                subtree = ID3(sub_X, sub_y, new_feature_names, feature_map, current_depth + 1, max_depth)
                tree[best_feature][value] = subtree

        return tree
    
# Function that trains a single tree and returns it
def train_single_tree(i, X_train, y_train, features, feature_map, max_depth):
    
    n_samples = X_train.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample = X_train[indices]
    y_sample = y_train[indices]

    # Train the tree
    current_depth = 0
    single_tree = ID3(X_sample, y_sample, features, feature_map, current_depth, max_depth)
    
    return single_tree, indices

# Function to calculate errors
def calculate_errors(forest, X_train, y_train, X_test, y_test, feature_map, most_common_label):
    train_predictions = bagged_trees_predict(forest, X_train, feature_map, most_common_label)
    train_error = np.mean(y_train != train_predictions)

    test_predictions = bagged_trees_predict(forest, X_test, feature_map, most_common_label)
    test_error = np.mean(y_test != test_predictions)

    return train_error, test_error


def bagging_stamp(X_train,y_train, features, feature_map, max_depth, tree_counts, X_test, y_test, most_common_label):
    # List to accumulate the forest; Lists to record the train and test errors
    bagged_forest = []
    train_errors = []
    test_errors = []

    # Use parallel computing to train trees
    results = Parallel(n_jobs=-1)(delayed(train_single_tree)(i, X_train, y_train, features, feature_map, max_depth)
                                  for i in range(1, tree_counts + 1))
    
    n_samples = X_train.shape[0]

    # Sequentially add trees to the forest and calculate cumulative errors
    for i in range(tree_counts):
        single_tree, indice = results[i]
        bagged_forest.append(single_tree)
        out_of_sample_indices = np.setdiff1d(np.arange(n_samples), indice)
        
        # Calculate errors with the current number of trees in the forest
        train_error, test_error = calculate_errors(bagged_forest, X_train[out_of_sample_indices], y_train[out_of_sample_indices], X_test, y_test, feature_map, most_common_label)
        
        # Record errors
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f"Trees: {i+1}, Train Error: {train_error}, Test Error: {test_error}")
    return train_errors, test_errors


#----------------------------------variance and bias
# Function that trains a single tree and returns it
def train_single_tree2(X_train, y_train, features, feature_map, max_depth):
    
    n_samples = X_train.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample = X_train[indices]
    y_sample = y_train[indices]

    # Train the tree
    current_depth = 0
    single_tree = ID3(X_sample, y_sample, features, feature_map, current_depth, max_depth)
    
    return single_tree

# Function to train bagged trees in parallel
def bagged_trees(X_train, y_train, features, num_trees, max_depth):
    feature_map = {feature: idx for idx, feature in enumerate(features)}
    
    # Train each tree in parallel
    trees = Parallel(n_jobs=-1)(delayed(train_single_tree2)(X_train, y_train, features, feature_map, max_depth) 
                                for _ in range(num_trees))
    return trees

# Function to compute bias and variance
def compute_bias_variance(predictions, true_labels):
    mean_predictions = np.mean(predictions, axis=0)  # E[h(x)]
    bias = np.mean((true_labels - mean_predictions) ** 2)  # f(x)
    variance = np.mean(np.var(predictions, axis=0))  # var(h(x))
    return bias, variance

# Main function for running iterations
def run_iteration_bagging(X_train, y_train, X_test, y_test, features, num_trees, max_depth, num_samples, feature_map, most_common_label):
    indices = np.random.choice(np.arange(X_train.shape[0]), size=num_samples, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]

    # Train bagged forest
    bagged_forest = bagged_trees(X_sample, y_sample, features, num_trees, max_depth)
    
    # Predict using the first tree (single tree learner)
    first_tree = bagged_forest[0]
    single_tree_predictions = predict(first_tree, X_test, feature_map, most_common_label)

    # Predict using the full bagged ensemble
    bagged_tree_predictions = bagged_trees_predict(bagged_forest, X_test, feature_map, most_common_label)

    return single_tree_predictions, bagged_tree_predictions

def plot_bagging(single_tree_bias, single_tree_variance, bagged_tree_bias, bagged_tree_variance):
    plt.figure(figsize=(8, 4))
    plt.bar(['Single Tree Bias', 'Single Tree Variance', 'Bagged Tree Bias', 'Bagged Tree Variance'],
                    [single_tree_bias, single_tree_variance, bagged_tree_bias, bagged_tree_variance])
    plt.ylabel('Error')
    plt.title('Bias and Variance Comparison')
    plt.show()

def plot_bagging_tree2(tree_counts,train_errors,test_errors):
    # Plot the errors for the current number of trees
    plt.plot([i for i in range(1, tree_counts+1)], train_errors, label='Train Error')
    plt.plot([i for i in range(1, tree_counts+1)], test_errors, label='Test Error')

    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title('Training and Test Errors vs Number of Trees (Bagged Trees)')
    plt.legend()
    plt.show()