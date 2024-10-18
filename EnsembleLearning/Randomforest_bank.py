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
def information_gain_fun(X, y, features, feature_map, feature_subset):
    total_entropy = entropy(X, y)

    best_info_gain = -np.inf
    best_feature = None

    for feature in feature_subset:
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
def predict(tree, X, feature_map, most_common_label, n_jobs=-1):
     
     return np.array(Parallel(n_jobs=n_jobs)(delayed(predict_one)(tree, x, feature_map, most_common_label) for x in X))

# Function to predict using the Random Forest
def random_forest_predict(trees, X, feature_map, most_common_label,n_jobs=-1):
    
    tree_predictions = np.array(Parallel(n_jobs=n_jobs)(delayed(predict)(tree, X, feature_map, most_common_label, n_jobs=1) for tree in trees))
    
    # Perform majority voting (aggregate predictions across trees)
    majority_vote = np.sign(np.sum(tree_predictions, axis=0))
    
    return majority_vote

# ID3 Algorithm with support for a user-defined max depth and safe feature sampling
def ID3(X, y, features, feature_map, current_depth, max_depth, num_features):
    if len(np.unique(y)) == 1:
        return y[0]
    
    # If there are no more features, or if the max depth is reached, return the most common label
    elif len(features) == 0 or (max_depth is not None and current_depth >= max_depth):
        shifted_y = y + 1  # Shift labels to ensure non-negative values for bincount
        most_common_label = np.bincount(shifted_y).argmax() - 1  # Shift back to original labels
        return most_common_label

    else:
        # Ensure num_features does not exceed the number of available features
        num_features = min(num_features, len(features))
        
        # Randomly select a subset of features for this split
        feature_subset = np.random.choice(features, size=num_features, replace=False)

        best_feature = information_gain_fun(X, y, features, feature_map, feature_subset)
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
                new_feature_names = [f for f in features if f != best_feature]
                subtree = ID3(sub_X, sub_y, new_feature_names, feature_map, current_depth + 1, max_depth, num_features)
                tree[best_feature][value] = subtree

        return tree


# Define the function to train a single tree
def train_single_tree(i, X_train, y_train, features, feature_map, max_depth, num_features):
    n_samples = X_train.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    current_depth = 0
    
    # Train a single tree with the specified number of features
    single_tree = ID3(X_sample, y_sample, features, feature_map, current_depth, max_depth, num_features)
    
    return single_tree, indices

# Function to calculate the error
def calculate_errors(forest, X_train, y_train, X_test, y_test, feature_map, most_common_label):
    # Training predictions and error
    train_predictions = random_forest_predict(forest, X_train, feature_map, most_common_label)
    train_error = np.mean(y_train != train_predictions)

    # Test predictions and error
    test_predictions = random_forest_predict(forest, X_test, feature_map, most_common_label)
    test_error = np.mean(y_test != test_predictions)

    return train_error, test_error


# compute bias and variance
def train_single_tree2(X_train, y_train, features, feature_map, max_depth,num_features):
    
    n_samples = X_train.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample = X_train[indices]
    y_sample = y_train[indices]

    # Train the tree
    current_depth = 0
    single_tree = ID3(X_sample, y_sample, features, feature_map, current_depth, max_depth,num_features)
    
    return single_tree

# Random Forest function with parallelized tree training
def random_forest(X, y, features, num_trees, max_depth, num_features):
    feature_map = {feature: idx for idx, feature in enumerate(features)}
    
    # Train each tree in parallel
    trees = Parallel(n_jobs=-1)(delayed(train_single_tree2)(X, y, features, feature_map, max_depth, num_features)
                                for _ in range(num_trees))
    return trees

# Function to compute bias and variance
def compute_bias_variance(predictions, true_labels):
    mean_predictions = np.mean(predictions, axis=0)  # E[h(x)]
    bias = np.mean((true_labels - mean_predictions) ** 2)  # Bias
    variance = np.mean(np.var(predictions, axis=0))  # Variance
    return bias, variance

# Main function to run an iteration
def run_iteration_rf(X_train, y_train, X_test, y_test, features, num_trees, max_depth, num_samples, num_features, feature_map, most_common_label):

    # Sample 1000 examples without replacement
    indices = np.random.choice(np.arange(X_train.shape[0]), size=num_samples, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]

    # Train a random forest
    rf = random_forest(X_sample, y_sample, features, num_trees, max_depth, num_features)
    
    # Predictions using the first tree (single tree learner)
    single_tree = rf[0]
    single_tree_predictions = predict(single_tree, X_test, feature_map, most_common_label)

    # Predictions using the full random forest
    bagged_tree_predictions = random_forest_predict(rf, X_test, feature_map, most_common_label)

    return single_tree_predictions, bagged_tree_predictions

def plot_rf_bias_variance (single_tree_bias_variance_squared_error, bagged_tree_bias_variance_squared_error ):
    # Labels for the x-axis
    labels = ['Single Tree Bias', 'Single Tree Variance', 'Forest Bias', 'Forest Variance']
    x = range(len(labels))

    # Data for plotting
    train_error_2_features = [
        single_tree_bias_variance_squared_error[0][0],
        single_tree_bias_variance_squared_error[0][1], 
        bagged_tree_bias_variance_squared_error[0][0],
        bagged_tree_bias_variance_squared_error[0][1] 
    ]

    train_error_4_features = [
        single_tree_bias_variance_squared_error[1][0],
        single_tree_bias_variance_squared_error[1][1], 
        bagged_tree_bias_variance_squared_error[1][0], 
        bagged_tree_bias_variance_squared_error[1][1]  
    ]

    train_error_6_features = [
        single_tree_bias_variance_squared_error[2][0],  # Single Tree Bias for features=6
        single_tree_bias_variance_squared_error[2][1],  # Single Tree Variance for features=6
        bagged_tree_bias_variance_squared_error[2][0],  # Forest Bias for features=6
        bagged_tree_bias_variance_squared_error[2][1]   # Forest Variance for features=6
    ]

    # Plotting the bars
    plt.figure(figsize=(10, 4))
    plt.bar(x, train_error_2_features, width=0.2, label='Train Error (features=2)', align='center')
    plt.bar([i + 0.2 for i in x], train_error_4_features, width=0.2, label='Train Error (features=4)', align='center')
    plt.bar([i + 0.4 for i in x], train_error_6_features, width=0.2, label='Train Error (features=6)', align='center')

    plt.ylabel('Error')
    plt.title('Bias and Variance Comparison (Single Tree vs Forest)')
    plt.xticks([i + 0.2 for i in x], labels)
    plt.legend()
    plt.show()
