
#upload package
import pandas as pd
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Seed for reproducibility
np.random.seed(0)

class PrimalSVM:
    def __init__(self, gamma, a, C, N):
        self.gamma = gamma
        self.a = a
        self.C = C
        self.N = N
        self.w = None  # Weight vector
        self.b = 0  # Bias term
        self.objective_curve = []  # Stores hinge loss at each epoch

    def _hinge_loss(self, X, y):
        loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b))) #objective function
        return loss

    def fit(self, X_train, y_train, epochs, schedule):
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)  # Initialize weights

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(n_samples)
            X_train, y_train = X_train[perm], y_train[perm]

            for i, (xi, yi) in enumerate(zip(X_train, y_train)):
                t = epoch * n_samples + i + 1  # Global step count

                # Learning rate schedule
                if schedule == "schedule1":
                    eta_t = self.gamma / (1 + (self.gamma / self.a) * t)
                elif schedule == "schedule2":
                    eta_t = self.gamma / (1 + t)
                else:
                    raise ValueError("Invalid schedule. Choose 'schedule1' or 'schedule2'.")

                # Sub-gradient updates
                if yi * (np.dot(self.w, xi) + self.b) <= 1:
                    self.w = (1 - eta_t) * self.w + eta_t * self.C * self.N * yi * xi
                    self.b += eta_t * self.C * yi
                else:
                    self.w = (1 - eta_t) * self.w

            # Compute hinge loss at the end of each epoch
            self.objective_curve.append(self._hinge_loss(X_train, y_train))
        

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)

    def get_objective_curve(self):
        return self.objective_curve
    
    def get_weights(self):
        return self.w
    
    def get_bias(self):
        return self.b

class DualSVM:
    def __init__(self, C):
        self.C = C  # Regularization parameter
        self.alpha = None  # Lagrange multipliers
        self.w = None  # Weight vector
        self.b = None  # Bias term


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # kernel matrix (linear kernel)
        K = np.dot(X, X.T)

        # define the dual objective function
        def objective(alpha):
            return -np.sum(alpha) + 0.5 * np.sum((alpha * y)[:, None] * (alpha * y) * K)

        # Initial guess for alpha
        alpha0 = np.zeros(n_samples)

        # Bounds for alpha: 0 <= alpha <= C
        bounds = [(0, C) for _ in range(n_samples)]

        # Equality constraint: sum(alpha * y) = 0
        constraints = {
            'type': 'eq',
            'fun': lambda alpha: np.dot(alpha, y),
            'jac': lambda alpha: y
            }
        
        # Solve the optimization problem
        result = minimize(
            objective,
            alpha0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # extract the optimal alpha
        self.alpha = result.x #minimize the function to get alpha

        # compute weight vector
        self.w = np.sum((self.alpha * y)[:, None] * X, axis=0)

        # compute bias term
        support_vector_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0][0]
        self.b = y[support_vector_idx] - np.dot(self.w, X[support_vector_idx])

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)
    
class GaussianSVM:
    def __init__(self, C, gamma):
        self.C = C
        self.gamma = gamma
        self.alpha = None  # lagrange multipliers
        self.b = None  # bias term
        self.X_train = None  # training features
        self.y_train = None  # training labels

    def gaussian_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / self.gamma)

    def kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.gaussian_kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples, n_features = X.shape

        # Compute the kernel matrix
        K = self.kernel_matrix(X)

        # Define the dual objective function
        def objective(alpha):
            return -np.sum(alpha) + 0.5 * np.sum((alpha * y)[:, None] * (alpha * y) * K)

        # Bounds for alpha: 0 <= alpha <= C
        bounds = [(0, self.C) for _ in range(n_samples)]

        # Equality constraint: sum(alpha * y) = 0
        constraints = {
            'type': 'eq',
            'fun': lambda alpha: np.dot(alpha, y),
            'jac': lambda alpha: y
        }

        # Initial guess for alpha
        alpha0 = np.zeros(n_samples)

        # Solve the optimization problem
        result = minimize(
            objective,
            alpha0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'disp': True}
        )

        # Extract the optimal alpha
        self.alpha = result.x

        # Compute bias term using support vectors
        support_vector_idx = np.where((self.alpha > 1e-4) & (self.alpha < self.C))[0]
        support_vector_idx = support_vector_idx[0]
        self.b = y[support_vector_idx] - np.sum(self.alpha * y * K[support_vector_idx])

    def predict(self, X):
        y_pred = []
        for x in X:
            # Decision function
            decision = np.sum(
                self.alpha * self.y_train *
                np.array([self.gaussian_kernel(x, x_train) for x_train in self.X_train])
            ) + self.b
            y_pred.append(np.sign(decision))
        return np.array(y_pred)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)  # Error rate

    def get_support_vectors(self):
        return np.where((self.alpha > 1e-4) & (self.alpha < self.C))[0]

class KernelPerceptron:
    def __init__(self, gamma, max_epochs=10):
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.c = None
        self.X_train = None
        self.y_train = None

    def gaussian_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2)**2 / self.gamma)

    def fit(self, X, y):
        N = X.shape[0]
        self.c = np.zeros(N)  # Mistake counts
        self.X_train = X
        self.y_train = y

        for epoch in range(self.max_epochs):
            for i in range(N):
                # Compute decision function
                decision = sum(self.c[j] * y[j] * self.gaussian_kernel(X[j], X[i]) for j in range(N))
                if y[i] * decision <= 0:  # Misclassified
                    self.c[i] += 1  # Update mistake count

    def predict(self, X):
        y_pred = []
        for x in X:
            # Compute decision function for new point
            decision = sum(self.c[i] * self.y_train[i] * self.gaussian_kernel(self.X_train[i], x) for i in range(len(self.X_train)))
            y_pred.append(np.sign(decision))
        return np.array(y_pred)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)  # Accuracy
