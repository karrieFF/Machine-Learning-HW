#upload package
import pandas as pd
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import SVMpkg #self-design package #why I unable to import the file in .py but I can import in jupyternotebook

np.random.seed(0) # Seed for reproducibility

#upload dataset
train_data = pd.read_csv("D:/EIC-Code/00-Python/Machine-Learning-HW/SVM/bank-note/train.csv", header = None, names = ['variance','skewness','curtosis','entropy','y'])
test_data = pd.read_csv("D:/EIC-Code/00-Python/Machine-Learning-HW/SVM/bank-note/test.csv", names = ['variance','skewness','curtosis','entropy','y'])

features = ['variance','skewness','curtosis','entropy']
outcome = 'y'

X_train = train_data[features].values #change to matrix multiple
y_train = train_data[outcome].values
X_test = test_data[features].values
y_test = test_data[outcome].values
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1


if __name__ == "__main__":
    # Define hyperparameter search space
    Cs = [100 / 873, 500 / 873, 700 / 873]
    gamma_values = [1, 0.5, 0.1, 0.01, 0.001]
    a_values = [1, 10, 50, 100]
    N = 1
    epochs = 100

    # Initialize variables to track the best parameters and lowest error for each schedule and C
    best_params_per_schedule = {"schedule1": {}, "schedule2": {}}

    # Perform grid search
    for schedule in ["schedule1", "schedule2"]:
        for C in Cs:
            lowest_error_for_C = float("inf")
            best_params_for_C = None
            for gamma in gamma_values:
                for a in a_values:
                    # Initialize and train the SVM
                    
                    svm = SVMpkg.PrimalSVM(gamma=gamma, a=a, C=C, N=N)
                    svm.fit(X_train, y_train, epochs=epochs, schedule=schedule)

                    # Calculate training and test errors
                    train_error = svm.score(X_train, y_train)
                    test_error = svm.score(X_test, y_test)
                    weights = svm.get_weights()
                    bias = svm.get_bias()

                    #print(f"Schedule: {schedule}, C={C:.6f}, gamma={gamma}, a={a}, "
                    #f"train_error: {train_error:.4f}, test_error: {test_error:.4f},"
                    #f"weights, {weights}, bias = {bias}")

                    #output objective curve 
                    #objective_curve = svm.get_objective_curve()
                    #plt.figure(figsize=(8, 5))
                    #plt.plot(range(1, len(objective_curve) + 1), objective_curve, marker='o')
                    #plt.title("Objective Function Curve (Hinge Loss)")
                    #plt.xlabel("Epoch")
                    #plt.ylabel("Hinge Loss")
                    #plt.grid(True)
                    #plt.show()

                    # Update the best parameters for the current C
                    if test_error < lowest_error_for_C:
                        lowest_error_for_C = test_error
                        best_params_for_C = {
                            "C": C,
                            "gamma": gamma,
                            "a": a,
                            "schedule": schedule,
                            "train_error": train_error,
                            "test_error": test_error,
                            "weights": weights,
                            "bias": bias
                        }
            
            # Store the best parameters for the current C and schedule
            best_params_per_schedule[schedule][C] = best_params_for_C

    # Print the best parameters for each C, separated by schedule
    for schedule, params_per_C in best_params_per_schedule.items():
        print(f"\nBest Parameters for {schedule}:")
        for C, params in params_per_C.items():
            print(f"C={C:.6f}, Schedule: {params['schedule']}, gamma={params['gamma']}, a={params['a']}, "
                f"train_error: {params['train_error']:.4f}, test_error: {params['test_error']:.4f}, "
                f"weights: {params['weights']}, bias: {params['bias']}")


if __name__ == "__main__":
    # Define hyperparameter C
    Cs = [100 / 873, 500 / 873, 700 / 873] #, 873

    # Train and evaluate the model for different values of C
    for C in Cs:
        svm = SVMpkg.DualSVM(C)
        svm.fit(X_train, y_train)
        train_error = svm.score(X_train, y_train)
        test_error = svm.score(X_test, y_test)
        print(f"C={C}, Train error={train_error:.2f}, Test error={test_error:.2f}")
        print(f"Weights:, {svm.w}, Bias:, {svm.b}")

    # Print the weights and bias for the best model
    print("Weights:", svm.w)
    print("Bias:", svm.b)

if __name__ == "__main__":
    Cs = [100 / 873, 500 / 873, 700 / 873] #, 873
    gammas = [0.1, 0.5, 1, 5, 100]

    lowest_error = float("inf")
    best_params = None

    for C in Cs:
        for gamma in gammas:
            gsvm = SVMpkg.GaussianSVM(C, gamma)
            gsvm.fit(X_train, y_train)
            sv_indices = gsvm.get_support_vectors()
            train_error = gsvm.score(X_train, y_train)
            test_error = gsvm.score(X_test, y_test)
            print(f"C={C:.4f}, gamma={gamma}, Train error={train_error:.4f}, Test error={test_error:.4f}, Number of Support Vectors: {len(sv_indices)}")

            if test_error < lowest_error:
                lowest_error = test_error
                best_params = (C, gamma)

    print(f"Best Parameters: C={best_params[0]:.4f}, gamma={best_params[1]}")
    print(f"Best Test Error: {lowest_error:.4f}")

if __name__=="__main__":
    # Hyperparameters
    C = 500 / 873 
    gammas = [0.01, 0.1, 0.5, 1, 5]

    # Track support vectors
    support_vectors = {}
    for gamma in gammas:
        gsvm = SVMpkg.GaussianSVM(C=C, gamma=gamma)
        gsvm.fit(X_train, y_train)
        sv_indices = gsvm.get_support_vectors()
        support_vectors[gamma] = sv_indices
        print(f"Gamma={gamma}, Number of Support Vectors: {len(sv_indices)}")

    # Calculate overlaps between consecutive gammas
    for i in range(len(gammas) - 1):
        gamma1, gamma2 = gammas[i], gammas[i + 1]
        overlap = len(np.intersect1d(support_vectors[gamma1], support_vectors[gamma2]))
        print(f"Overlap between gamma={gamma1} and gamma={gamma2}: {overlap}")

if __name__=="__main__":
    # Test different gamma values
    gammas = [0.1, 0.5, 1, 5, 100]
    for gamma in gammas:
        kp = SVMpkg.KernelPerceptron(gamma=gamma)
        kp.fit(X_train, y_train)
        train_error = kp.score(X_train, y_train)
        test_error = kp.score(X_test, y_test)
        print(f"Gamma={gamma}, Train error ={train_error:.4f}, Test error ={test_error:.4f}")
