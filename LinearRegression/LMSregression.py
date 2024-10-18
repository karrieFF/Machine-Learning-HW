import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data

train_data = pd.read_csv("D:\\EIC-Code\\00-Python\\Machine-Learning-HW\\LinearRegression\\concrete\\concrete\\test.csv", header=None,
names = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr','y']) #y is SLUMP

test_data = pd.read_csv("D:\\EIC-Code\\00-Python\\Machine-Learning-HW\\LinearRegression\\concrete\\concrete\\test.csv", header=None, 
names = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr','y'])

features = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']

X_train = train_data[features].values
y_train = train_data['y'].values
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# Define the cost function (Mean Squared Error)
def compute_cost(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    cost = 1/2 * np.sum((predictions - y) ** 2) #* m
    return cost

# Define the gradient of the cost function
def compute_batch(X, y, w):
    prediction = np.dot(X, w)
    error = y - prediction
    gradient = - np.dot(X.T, error)#X.T.dot(error)  # Gradient for all sample
    return gradient

# Function to compute gradient for a single sample
def compute_stochastic(Xi, yi, w):
    prediction = np.dot(Xi, w)
    error = yi - prediction
    gradient = - error * Xi  # Gradient for one sample
    return gradient

# Batch Gradient Descent implementation
def batch_gradient_descent(X, y, r, tolerance=1e-6, decay_rate = 0.01, epoches=1000):
    m, n = X.shape
    w = np.zeros(n)  # Initialize weights to zeros
    cost_history = []
    r_history1 = []
    w_diff = np.inf  # Set initial weight difference to infinity
    
    for t in range(epoches):
        # Compute the current cost
        cost = compute_cost(X, y, w)
        cost_history.append(cost)
        
        # Compute the gradient
        grad = compute_batch(X, y, w)

        # Update weights
        w_new = w - r * grad
        
        # Calculate the difference in weights
        w_diff = np.linalg.norm(w_new - w)
        
        # Update weights for the next iteration
        w = w_new

        #update learning rate
        r = r / (1 + decay_rate * 2)
        r_history1.append(r)

        # Monitor learning rate and convergence
        if w_diff < tolerance:
            print(f"Convergence reached at iteration {t}")
            break
    
    return w, cost_history, r_history1


# Stochastic Gradient Descent (SGD) implementation
def stochastic_gradient_descent(X, y, r, epoches=1000, decay_rate = 0.01, tolerance=1e-6):
    m, n = X.shape
    w = np.zeros(n)  # Initialize weights to zero
    cost_history = []
    r_history2 =[ ]
    w_diff = np.inf  # Set initial weight difference to infinity
    
    for t in range(epoches):
        # Randomly sample an index
        i = np.random.randint(m)
        
        # Get the randomly selected sample
        Xi = X[i]
        yi = y[i]
        
        # Compute gradient for this sample
        dw = compute_stochastic(Xi, yi, w)

        # Update weights
        w_new = w - r * dw

        # Calculate weight difference
        w_diff = np.linalg.norm(w_new - w)
        w = w_new  # Update weights for next iteration

        # Compute and store the cost function (on the whole dataset) after each update
        cost = compute_cost(X, y, w)
        cost_history.append(cost)

        #update learning rate
        r = r / (1 + decay_rate * 2)
        r_history2.append(r)
        
        # Check for convergence
        if w_diff < tolerance:
            print(f"Convergence reached at iteration {t}")
            break
    
    return w, cost_history, r_history2

def analytical_solution(X, y):
    w_star = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w_star

#-----batch_gradient_descent
if __name__ == "__main__":

    # Run the batch gradient descent algorithm
    learning_rate1 = 0.0001 
    weights, cost_history, r_history1 = batch_gradient_descent(X_train_bias, y_train, r=learning_rate1)

    # Plot the cost function over iterations
    plt.plot(cost_history)
    plt.title("Cost Function over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # After convergence, print the final weight vector and the final learning rate
    print(f"Final weights: {weights}")
    print(f"Final learning rate: {r_history1[-1]}")

    # Use the final weight vector to compute the cost function value on the test data
    X_test = test_data[features].values
    y_test = test_data['y'].values
    X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    test_cost = compute_cost(X_test_bias, y_test, weights)
    print(f"Cost on test data: {test_cost}")

#-----stochastic_gradient_descent
if __name__ == "__main__":
    # Run the Stochastic Gradient Descent algorithm
    learning_rate2 = 0.01
    weights, cost_history, r_history2 = stochastic_gradient_descent(X_train_bias, y_train, r=learning_rate2)

    # Plot the cost function over iterations
    plt.plot(cost_history)
    plt.title("Cost Function over Updates")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # After convergence, print the final weight vector and the final cost
    print(f"final weights: {weights}")
    print(f"final learning_rate: {r_history2[-1]}")
    print(f"final cost on training data: {cost_history[-1]}")

    # Use the final weights to compute the cost on the test data
    X_test = test_data[features].values
    y_test = test_data['y'].values
    X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    test_cost = compute_cost(X_test_bias, y_test, weights)
    print(f"Cost on test data: {test_cost}")


# Compute optimal weights using analytic approach
if __name__ == "__main__":
    optimal_weights = analytical_solution(X_train_bias, y_train)
    print(f"optimal_weights: {optimal_weights}")