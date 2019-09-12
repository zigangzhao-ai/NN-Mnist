import os
import numpy as np
import random
import matplotlib.pyplot as plt 

from activations import sigmoid, sigmoid_prime


class NeuralNetwork(object):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16,
                 epochs=10):
        """Initialize a Neural Network model.

        Parameters
        ----------
        sizes : list, optional
            A list of integers specifying number of neurns in each layer. Not
            required if a pretrained model is used.

        learning_rate : float, optional
            Learning rate for gradient descent optimization. Defaults to 1.0

        mini_batch_size : int, optional
            Size of each mini batch of training examples as used by Stochastic
            Gradient Descent. Denotes after how many examples the weights
            and biases would be updated. Default size is 16.

        """
        # Input layer is layer 0, followed by hidden layers layer 1, 2, 3...
        self.sizes = sizes
        self.num_layers = len(sizes)

        # First term corresponds to layer 0 (input layer). No weights enter the
        # input layer and hence self.weights[0] is redundant.
        self.weights = [np.array([0])] + [np.random.randn(y, x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = learning_rate
        

        

    def fit(self, training_data, test_data=None):
        """Fit (train) the Neural Network on provided training data. Fitting is
        carried out using Stochastic Gradient Descent Algorithm.

        Parameters
        ----------
        training_data : list of tuple
            A list of tuples of numpy arrays, ordered as (image, label).

        validation_data : list of tuple, optional
            Same as `training_data`, if provided, the network will display
            validation accuracy after each epoch.
            
        """
       
        training_data = list(training_data)
        test_data = list(test_data)
        
        n = len(test_data)
        print("the number of test_data is :",n)
        
        
        error_list = [] 
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [
                    w - (self.eta / self.mini_batch_size) * dw for w, dw in
                    zip(self.weights, nabla_w)]
                self.biases = [
                    b - (self.eta / self.mini_batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]
                
                #train_results = [(np.argmax(self.predict(x)), y) for (x, y) in training_data]
                #y1=[np.argmax(self.predict(x)) for x in training_data]
                #y =training_data[1]
                #loss = self.cross_entropy_error(y1, y)
         
            if test_data:
                
                accuracy = self.validate(test_data) / 100.0
                
                error_list.append(100-accuracy)
                #print(error_list)
                
                print("Epoch{}: {} / {}".format(epoch+1,self.validate(test_data),n))
                print("Epoch{0}: test accuracy is {1}%.".format(epoch + 1, accuracy))
                #print('Epoch:', epoch+1, '| loss: %.2f' % loss)

                
           
            else:
                print("Processed epoch {0}.".format(epoch))
                
        plt.figure()
        
            
        x1 = range(0,self.epochs)
        print(x1)
        y1 = error_list
        #print(error_list)
        plt.plot(x1, y1, 'o-')
        plt.title('error_curves vs epoches')
        plt.ylabel('error_rate')
        plt.show()
                

    def validate(self, test_data):
        """Validate the Neural Network on provided validation data. It uses the
        number of correctly predicted examples as validation accuracy metric.

        Parameters
        ----------
        validation_data : list of tuple

        Returns
        -------
        int
            Number of correctly predicted images.

        """
        validation_results = [(self.predict(x) == y) for x, y in test_data]
        return sum(result for result in validation_results)

    def predict(self, x):
        """Predict the label of a single test example (image).

        Parameters
        ----------
        x : numpy.array

        Returns
        -------
        int
            Predicted label of example (image).

        """

        self._forward_prop(x)
        return np.argmax(self._activations[-1])

    def _forward_prop(self, x):
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])

    def _back_prop(self, x, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y) * sigmoid_prime(self._zs[-1])
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                sigmoid_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w
        
        
    
    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

            batch_size = y.shape[0]
            delta = 1e-7

        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size















    def load(self, filename='model.npz'):
        """Prepare a neural network from a compressed binary containing weights
        and biases arrays. Size of layers are derived from dimensions of
        numpy arrays.

        Parameters
        ----------
        filename : str, optional
            Name of the ``.npz`` compressed binary in models directory.

        """
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

        # Bias vectors of each layer has same length as the number of neurons
        # in that layer. So we can build `sizes` through biases vectors.
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        # These are declared as per desired shape.
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        # Other hyperparameters are set as specified in model. These were cast
        # to numpy arrays for saving in the compressed binary.
        self.mini_batch_size = int(npz_members['mini_batch_size'])
        self.epochs = int(npz_members['epochs'])
        self.eta = float(npz_members['eta'])

    def save(self, filename='model.npz'):
        """Save weights, biases and hyperparameters of neural network to a
        compressed binary. This ``.npz`` binary is saved in 'models' directory.

        Parameters
        ----------
        filename : str, optional
            Name of the ``.npz`` compressed binary in to be saved.

        """
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            mini_batch_size=self.mini_batch_size,
            epochs=self.epochs,
            eta=self.eta
        )

