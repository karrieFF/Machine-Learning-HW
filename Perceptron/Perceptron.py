import pandas as pd
import numpy as np
import time

#upload dataset
train_data = pd.read_csv("D:/EIC-Code/00-Python/Machine-Learning-HW/Perceptron/bank-note/train.csv", header = None, names = ['variance','skewness','curtosis','entropy','y'])
test_data = pd.read_csv("D:/EIC-Code/00-Python/Machine-Learning-HW/Perceptron/bank-note/test.csv", names = ['variance','skewness','curtosis','entropy','y'])

features = ['variance','skewness','curtosis','entropy']
outcome = 'y'

X_train = train_data[features].values #change to matrix multiple
y_train = train_data[outcome].values
X_test = test_data[features].values
y_test = test_data[outcome].values
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1


class StandardPerceptron: #only when you have class, you will need self. A class is a blueprint for creating objects. It defines a set of attributes and methods that the objects created from it will have.
    def __init__(self, max_epoches, learning_rate):
        self.max_epoches = max_epoches
        self.weights = None
        self.r = learning_rate
    
    def fit(self, X, y):
        start_time = time.time()
        n = X.shape[1] # number of features
        self.weights = np.zeros(n) #inital weights

        for epoch in range(self.max_epoches):  
            
            # Set the random seed
            #np.random.seed(42)
            #shuffle the data
            data = np.column_stack((X, y))
            np.random.shuffle(data)
            X_shuffled = data[:, :-1]
            y_shuffled = data[:, -1]

            for i in range(len(X)):
                y_predict = np.sign(X_shuffled[i].dot(self.weights)) #Shuffle the data, how to shuffle the data
                test = y_predict*y_shuffled[i]
                if test <= 0:
                    self.weights  += self.r*y_shuffled[i]*X_shuffled[i] #wt+1 ←wt + r (yi xi)
                    
        end_time = time.time()
        fit_time = end_time - start_time
        print(f"Training time: {fit_time:.4f} s")
        
    def predict(self, X): #we need to use self anytime 
        y_predict = np.sign(X.dot(self.weights))
        return y_predict
    
    def scores(self, X, y):
        predictions = self.predict(X) 
        errors = predictions != y
        average_prediction_errors = np.mean(errors) #calculate the mean of true value
        return average_prediction_errors
    

if __name__ == "__main__":

    #set hyperparameters
    epoches = 10
    learning_rate =  0.1 #[0.01, 0.05, 0.1, 1] #learning rate is a small positive number less than or equal to 1, do I need to set the learning rate here?

    #for lr in learning_rate:
        #fit model
    SPmodel = StandardPerceptron(epoches, learning_rate)
    SPmodel.fit(X_train, y_train)

        #return weights and model performance
    weights = SPmodel.weights
    average_score = SPmodel.scores(X_test, y_test)
    print('standard_perceptron','learning rate:', learning_rate, 'final weights:', weights, 'model performance:', average_score)


class VotePerceptron: #only when you have class, you will need self.
    def __init__(self, max_epoches, learning_rate):
        self.max_epoches = max_epoches
        self.r = learning_rate
        self.weights = []
        self.c = []
    
    def fit(self, X, y):
        start_time = time.time()
        n = X.shape[1] # number of features
        w = np.zeros(n) #inital weights
        m = 0 
        c = 0

        for epoch in range(self.max_epoches):
            for i in range(len(X)):
                y_predict = X[i].dot(w)
                test = y[i]*y_predict
                if test <= 0: # for misclassification
                    if c > 0:
                        self.c.append(c)
                        self.weights.append(w.copy())
                    w += self.r*y[i]*X[i]
                    m += 1 
                    c = 1 #if these is one missclassification, we will reset the count
                else:
                    c += 1 #c is the count of correc predictions

        #print(len(self.weights), len(self.c))
        end_time = time.time()
        fit_time = end_time - start_time
        print(f"Training time: {fit_time:.4f} s")

    def predict(self, X): #we need to use self anytime
        predictions = []
        for i in range(len(X)):
            final_predict = 0
            for ci, wi in zip(self.c, self.weights):
                one_predict = np.sign(np.dot(wi, X[i]))
                sum_ci_ypredict = ci * one_predict
                final_predict += sum_ci_ypredict
            predictions.append(np.sign(final_predict))
        return predictions

    def scores(self, X, y):
        predictions = self.predict(X)
        errors = predictions != y
        average_prediction_errors = np.mean(errors)
        return average_prediction_errors
    

if __name__ == "__main__":
    max_epoches = 10
    learning_rate = 0.1 #[0.01, 0.05, 0.1, 1]
    
    #for lr in learning_rate:
    # #define the model
    
    votemodel = VotePerceptron(max_epoches, learning_rate)
    
    votemodel.fit(X_train, y_train)

    distinct_weights = votemodel.weights
    distinct_count = votemodel.c

    model_performance = votemodel.scores(X_test, y_test)
    print('vote_perceptron')
    print('learning rate:', learning_rate)
    print("the list of the distinct weight and count:", [i for i in zip(distinct_weights, distinct_count)])
    print('model performance:', model_performance)

class averageperceptron: #only when you have class, you will need self.
    def __init__(self, max_epoches, learning_rate):
        self.max_epoches = max_epoches
        self.r = learning_rate
        self.weights = None
        self.a = None
    
    def fit(self, X, y):
        start_time = time.time()
        n = X.shape[1] # number of features
        self.weights = np.zeros(n) #inital weights
        self.a = np.zeros(n)

        for epoch in range(self.max_epoches):
            for i in range(len(X)):
                y_predict = X[i].dot(self.weights)
                test = y[i]*y_predict
                if test <= 0: # for misclassification
                    self.weights += self.r*y[i]*X[i]
                self.a += self.weights #store the accurate weights
        end_time = time.time()
        fit_time = end_time - start_time
        print(f"Training time: {fit_time:.4f} s")


    def predict(self, X): #we need to use self anytime
        predictions = []
        for i in range(len(X)):
            prediction = np.sign(np.dot(X[i], self.a))
            predictions.append(prediction)
        
        return predictions

    def scores(self, X, y):
        predictions = self.predict(X)
        errors = predictions != y
        average_prediction_errors = np.mean(errors)
        return average_prediction_errors

if __name__ == "__main__":
    max_epoches = 10
    learning_rate = 0.1 #[0.01, 0.05, 0.1, 1]

    #for lr in learning_rate:
    # #define the model
    votemodel = averageperceptron(max_epoches, learning_rate)
    votemodel.fit(X_train, y_train)

    distinct_weights = votemodel.weights
    model_performance = votemodel.scores(X_test, y_test)

    print('averaged_perceptron', 'learning rate:', learning_rate, "the final weights:", distinct_weights, 'model performance:', model_performance)