This is a machine learning library developed by Lingyi Fu (u1452585) for CS5350/6350 in University of Utah


### How to Run the algorithms?

Please run "main_bank.py" file in the EnsembleLearning folder for the Adaboost, Bagging trees, Random Forest algorithm for the Bank dataset.

Please run "credit.py" file in the EnsembleLearning folder for the Adaboost, Bagging trees, Random forest algorithm for the credit dataset.

Please run "LMSregression.py" file in the LinearRegression folder for the LMS with batch-gradient and stochastic gradient.

Please run the "Perceptron.py" file in the Perceptron folder for the Standard Perceptron/Voted Perceptron/Averaged Perceptron.

Please run the "main.ipynb" file in the SVM folder for the SVM with Primal form, Dual form, Gaussian kernel, and kernel perceptron algorithm.

Please run the "Neuronetwork.ipynb" file in the NeuralNetworks folder for the NeuralNetwork algorithm.

Please run the "logisticregression.ipynb" file in the Logisticregression folder for the logistic regression with MAP estimation and ML estimation.


### How to set the hyperparameters?

1. Adaboost
features: features of the dataset 
num_iterations: the number of epoches that will be runed to update the weights #500

2. Bagging
features: features of the dataset
tree_counts = number of trees #500
max_depth = the depth of each tree #len(features) 

3. Random forest
features: features of the dataset
tree_counts: nuber of trees #500
feature_subsets: random subset of features #[2, 4, 6] 
max_depth: the depth of each tree #len(features)

4. LMS with batch-gradient 
learning_rate1: the step size at each iteration  #0.0001 

5. Stochastic gradient
learning_rate1: the step size at each iteration  #0.0001 

6. Standard perceptron/Voted perceptron/Averaged perceptron
epoches: number of iterations
learning_rate: the spead of learning

7. SVM - PrimalSVM
Cs: regularization parameter
gamma_values: learning rate, te spead of learning
a_values: parametrs for calculating "schedule of learning rate"
N: Number of training samples, if it is Stochastic regression, each iteration is 1.
epochs: number of iterations

8. SVM - DualSVM
Cs: regularization parameter

9. SVM - Gaussian SVM
Cs: regularization parameter
gammas: learning rate

10. SVM - Kernel Perceptron
gammas: learning rate

11. Neural Networks
widths: width of hidden layer
depths: the number of hidden layers
gamma_0:  learning rate
d: hyperparmeters for calculating scheduled learning rate
epochs: number of iterations
activation_functions: different active function ("tanh", "relu") for introducing non-linearity into a neural network.

12. Logistic regression with MAP estimation
variances: hyperparmeters for calculating MAP estimation
gamma_0_values: learning rate
d_values: hyperparmeters for calculating scheduled learning rate
epochs: number of iterations

12. Logistic regression with ME estimation

gamma_0_values: learning rate
d_values: hyperparmeters for calculating scheduled learning rate
epochs: number of iterations



