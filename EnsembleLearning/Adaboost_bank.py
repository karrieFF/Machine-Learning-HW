import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate entropy for any label type, considering weights
def entropy(X, y, weights):
    total_weights = np.sum(weights)
    class_labels, class_weights = np.unique(y, return_inverse=True)
    weighted_counts = np.bincount(class_weights, weights=weights)
    probabilities = weighted_counts / total_weights
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding epsilon for stability
    return entropy_value

# Function to calculate information gain for each feature
def information_gain_fun(X, y, weights, feature_names, feature_map):
    total_entropy = entropy(X, y, weights)

    best_info_gain = -np.inf
    best_feature = None

    for feature in feature_names:
        feature_idx = feature_map[feature]
        expected_entropy = 0
        feature_values = X[:, feature_idx]  # Access column by index (mapped from feature name)
        unique_values = np.unique(feature_values)

        for value in unique_values:
            sub_mask = feature_values == value
            sub_X, sub_y, sub_weights = X[sub_mask], y[sub_mask], weights[sub_mask]
            sub_entropy = entropy(sub_X, sub_y, sub_weights)
            proportion = np.sum(sub_weights) / np.sum(weights)
            expected_entropy += proportion * sub_entropy

        information_gain = total_entropy - expected_entropy
        if information_gain > best_info_gain:
            best_info_gain = information_gain
            best_feature = feature

    return best_feature

# ID3 Algorithm with weights and max depth using np.array inputs
def ID3(X, y, weights, feature_names, feature_map, current_depth, max_depth):
    # If all target labels are the same, return the label
    if len(np.unique(y)) == 1:
        return y[0]

    # If no more features or max depth reached, return the most common label
    elif len(feature_names) == 0 or (current_depth >= max_depth):
        shifted_y = y + 1  # Shift y to ensure non-negative values for bincount
        most_common_label_idx = np.bincount(shifted_y, weights=weights).argmax()
        return most_common_label_idx - 1  # Shift back to original values

    else:
        best_feature = information_gain_fun(X, y, weights, feature_names, feature_map)
        tree = {best_feature: {}}

        best_feature_idx = feature_map[best_feature]

        for value in np.unique(X[:, best_feature_idx]):
            sub_mask = X[:, best_feature_idx] == value
            sub_X, sub_y, sub_weights = X[sub_mask], y[sub_mask], weights[sub_mask]

            if sub_X.shape[0] == 0:
                shifted_y = y + 1  # Shift y to ensure non-negative values for bincount
                most_common_label_idx = np.bincount(shifted_y, weights=weights).argmax()
                tree[best_feature][value] = most_common_label_idx - 1  # Shift back to original values
            else:
                new_feature_names = [f for f in feature_names if f != best_feature]
                subtree = ID3(sub_X, sub_y, sub_weights, new_feature_names, feature_map, current_depth + 1, max_depth)
                tree[best_feature][value] = subtree

        return tree

# Function to predict the label for one sample
def predict_one(tree, x, feature_map):
    if not isinstance(tree, dict):
        return tree

    parent_node = next(iter(tree))  # This is the feature name (string)
    subtree = tree[parent_node]
    feature_idx = feature_map[parent_node]
    node_value = x[feature_idx]  # Access the feature value based on the feature index

    if node_value in subtree:
        return predict_one(subtree[node_value], x, feature_map)
    else:
        return None

# Function to predict labels for a dataset
def predict(tree, X, feature_map):
    predict_values = []
    for x in X:  # Iterate over numpy array rows
        predict_value = predict_one(tree, x, feature_map)
        predict_values.append(predict_value)
    return np.array(predict_values)

# Function to calculate weighted error
def calculate_weighted_error(y_true, y_pred, weights):
    incorrect = (y_true != y_pred)
    error = np.sum(weights[incorrect]) / np.sum(weights)
    return error

# Function to make predictions using the AdaBoost ensemble
def adaboost_predict(weak_learners, alphas, X, feature_map):
    weighted_sum = np.zeros(X.shape[0])
    
    for tree, alpha in zip(weak_learners, alphas):
        y_pred = predict(tree, X, feature_map)
        weighted_sum += alpha * y_pred
    
    return np.sign(weighted_sum)

# Decision Stump: Modified ID3 algorithm for depth 1 (decision stump)
def decision_stump(X, y, weights, feature_names, feature_map):
    best_feature = information_gain_fun(X, y, weights, feature_names, feature_map)
    tree = {best_feature: {}}
    
    best_feature_idx = feature_map[best_feature]
    
    # Create leaf nodes for the two possible splits (binary decision)
    for value in np.unique(X[:, best_feature_idx]):
        sub_mask = X[:, best_feature_idx] == value
        sub_y = y[sub_mask]
        
        # Create leaf node: most common label
        if len(sub_y) == 0:
            continue
        tree[best_feature][value] = np.sign(np.sum(weights[sub_mask] * sub_y))
    
    return tree

# AdaBoost with Decision Stumps
def adaboost_stumps(X, y, features, num_iterations, X_test, y_test):
    n_samples = X.shape[0]
    n_test_samples = X_test.shape[0]
    
    # Initialize sample weights to be uniform
    weights = np.ones(n_samples) / n_samples
    weak_learners = []
    alphas = []
    
    train_errors = []
    test_errors = []
    stump_train_errors = []
    stump_test_errors = []
    
    feature_map = {feature: idx for idx, feature in enumerate(features)}
    
    for t in range(1, num_iterations + 1):
        # Train a weak learner (decision stump)
        stump = decision_stump(X, y, weights, features, feature_map)
        weak_learners.append(stump)
        
        # Predict on training data
        y_pred_train = predict(stump, X, feature_map)
        y_pred_test = predict(stump, X_test, feature_map)
        
        # Calculate weighted error
        error_t = calculate_weighted_error(y, y_pred_train, weights)
        
        # Compute the weight of the weak learner
        alpha_t = 0.5 * np.log((1 - error_t) / (error_t + 1e-10))  # Avoid division by zero
        alphas.append(alpha_t)
        
        # Update sample weights
        weights *= np.exp(-alpha_t * y * y_pred_train)  # Update rule for AdaBoost
        weights /= np.sum(weights)  # Normalize weights
        
        # Calculate stump errors for this iteration
        stump_train_error = np.mean(y != y_pred_train)
        stump_test_error = np.mean(y_test != y_pred_test)
        stump_train_errors.append(stump_train_error)
        stump_test_errors.append(stump_test_error)
        
        # Predict with the current ensemble on training and test data
        ensemble_train_pred = adaboost_predict(weak_learners, alphas, X, feature_map)
        ensemble_test_pred = adaboost_predict(weak_learners, alphas, X_test, feature_map)
        
        # Calculate training and test errors for the ensemble
        train_error = np.mean(y != ensemble_train_pred)
        test_error = np.mean(y_test != ensemble_test_pred)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    return train_errors, test_errors, stump_train_errors, stump_test_errors


# Function to plot the results
def plot_results(train_errors, test_errors, stump_train_errors, stump_test_errors, num_iterations):
    # Plot 1: Training and Test Errors vs. Number of Iterations
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_iterations + 1), train_errors, label='All Stumps Training Error')
    plt.plot(range(1, num_iterations + 1), test_errors, label='All stumps Test Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Training and Test Errors vs. Number of Iterations')
    plt.legend()

    # Plot 2: Stump Training and Test Errors per Iteration
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_iterations + 1), stump_train_errors, label='Stump Training Error')
    plt.plot(range(1, num_iterations + 1), stump_test_errors, label='Stump Test Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Stump Training and Test Errors per Iteration')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_bagging(single_tree_bias, single_tree_variance, bagged_tree_bias, bagged_tree_variance):
    plt.figure(figsize=(8, 4))
    plt.bar(['Single Tree Bias', 'Single Tree Variance', 'Bagged Tree Bias', 'Bagged Tree Variance'],
                    [single_tree_bias, single_tree_variance, bagged_tree_bias, bagged_tree_variance])
    plt.ylabel('Error')
    plt.title('Bias and Variance Comparison')
    plt.show()