import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1+np.exp(-x))

def ReLu(x):
    return np.maximum(x, 0)

def purelin(x):
    return x

def hardlim(x):
    return np.asarray(x>0).astype(int)

def hardlims(x):
    return np.asarray((x>0)*2-1).astype(int)

def tanh(x):
    return 2 / (1+np.exp(-2*x)) - 1

class Neuron():

    def __init__(self, n_inputs=9, bias=None, weights=None, n_epochs=30):
        self.n_inputs = n_inputs
        self.activation_function = hardlim
        if weights is None:
            self.weights = np.random.normal(loc=0, scale=0.01, size=[self.n_inputs])
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.random.normal(loc=0, scale=0.01)
        else:
            self.bias = bias
        self.n_epochs = n_epochs
        self.weight_updates, self.bias_updates = [], []
        self.mse, self.accuracy = [], []

    def output(self, input):
        x = np.sum(input*self.weights) + self.bias
        return self.activation_function(x)

    def classify(self, inputs):
        y_pred = []
        for input in inputs:
            pred = self.output(input)
            y_pred.append(pred)
        return np.array(y_pred)

    def evaluate_accuracy(self, inputs, targets):
        y_pred = self.classify(inputs)
        return accuracy_score(targets, y_pred)*100

    def train(self, inputs, targets, learning_rate=0.01, n_epochs=None, plot=False,
              store_updates=True, verbose=True):
        if n_epochs is None:
            n_epochs = self.n_epochs

        # mse = self.MSE(inputs, targets)
        # acc = self.evaluate_accuracy(inputs, targets)
        # self.mse.append(mse)
        # self.accuracy.append(acc)
        # print('Before training: MSE: %s - Training accuracy: %s'
        #       % (self.mse[-1], self.accuracy[-1]))

        for i in range(n_epochs):
            
            for pattern, target in zip(inputs, targets):
                err = self.output(pattern) - target
                self.weights = self.weights - learning_rate * err * pattern
                self.bias    = self.bias - learning_rate * err
            if store_updates:
                self.weight_updates.append(self.weights)
                self.bias_updates.append(self.bias)
            self.mse.append(self.MSE(inputs, targets))
            acc = self.evaluate_accuracy(inputs, targets)
            self.accuracy.append(acc)
            if verbose:
                print('--- Epoch: %s - MSE: %s - Training accuracy: %s'
                      %(i+1, self.mse[-1], acc))

    def MSE(self, inputs, targets):
        se = []
        for pattern, target in zip(inputs, targets):
            se.append(np.square(target - self.output(pattern)))
        return np.mean(se)


    def plot_weights(self):
        f, ax = plt.subplots(1, 1, figsize=[5,5])
        sns.heatmap(self.weights.reshape((3, 3)),
                    square=True,
                    annot=np.arange(9).reshape((3, 3)))
        ax.set_title('Weights')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()


    def plot_error_surface(self, X, y, min=-2, max=2, step=0.1):

        weights_grid, bias_grid = np.mgrid[min:max:step, min:max:step]
        MSE_grid = np.zeros_like(weights_grid)
        for pattern, target in zip(X, y):
            MSE_grid += np.square(weights_grid * pattern + bias_grid - target)

        MSE_grid = MSE_grid / X.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(weights_grid, bias_grid, MSE_grid, cmap="autumn_r",
                        lw=0.5, rstride=1, cstride=1, alpha=0.5)
        ax.contour(weights_grid, bias_grid, MSE_grid, 10, lw=3,
                   cmap="autumn_r", linestyles="solid", offset=-1)
        ax.contour(weights_grid, bias_grid, MSE_grid, 10, lw=3, colors="k",
                   linestyles="solid")
        ax.set_xlabel('Weight')
        ax.set_ylabel('Bias')
        try:
            w = np.array(self.weight_updates)
            b = np.array(self.bias_updates)
            m = np.array(self.mse)
            ax.plot(w, b, m, c='b')
            ax.scatter(w, b, m, s=40, c='b')
        except AttributeError:
            pass


class TrainingSet():

    def __init__(self, patterns, targets):
        self.patterns = patterns
        self.targets = targets

    def plot(self):
        f, ax = plt.subplots(3, 2, figsize=[7, 9])
        i = 0
        for axis in ax.flatten():
            sns.heatmap(self.patterns[i,:].reshape((3, 3)),
                        annot=np.arange(9).reshape((3, 3)),
                        vmin=self.patterns.min(),
                        vmax=self.patterns.max(),
                        ax=axis
                        )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title('Target: %s' % self.targets[i])
            i+=1
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return 'Training set containing '+str(self.patterns.shape[0]) + \
               ' patterns with associated target outputs.'


letter_A = np.array([[-1, -1, -1, -1, 1, -1, -1, -1, -1],
                 [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                 [-1, -1, 1, 1, 1, 1, 1, -1, -1],
                 [-1, -1, 1, 1, -1, 1, 1, -1, -1],
                 [-1, 1, 1, -1, -1, -1, 1, 1, -1],
                 [-1, 1, 1, 1, 1, 1, 1, 1, -1],
                 [-1, 1, 1, -1, -1, -1, 1, 1, -1],
                 [-1, 1, 1, -1, -1, -1, 1, 1, -1],
                 [-1, -1, -1, -1, -1, -1, -1, -1, -1]])

letter_A_noisy = letter_A.copy()
letter_A_noisy[:, 5:] = np.random.choice([1, -1], size=[9, 4])

letter_B = np.array([[-1, 1, 1, 1, 1, 1, 1, -1, -1],
                 [-1, 1, 1, 1, 1, 1, 1, 1, -1],
                 [-1, 1, 1, -1, -1, -1, 1, 1, -1],
                 [-1, 1, 1, -1, -1, -1, 1, 1, -1],
                 [-1, 1, 1, 1, 1, 1, 1, -1, -1],
                 [-1, 1, 1, -1, -1, -1, 1, 1, -1],
                 [-1, 1, 1, -1, -1, -1, 1, 1, -1],
                 [-1, 1, 1, 1, 1, 1, 1, 1, -1],
                 [-1, 1, 1, 1, 1, 1, 1, -1, -1]])


letter_C = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                 [-1, -1, 1, 1, 1, 1, 1, -1, -1],
                 [-1, 1, 1, 1, 1, 1, 1, -1, -1],
                 [-1, 1, 1, -1, -1, -1, -1, -1, -1],
                 [-1, 1, 1, -1, -1, -1, -1, -1, -1],
                 [-1, 1, 1, -1, -1, -1, -1, -1, -1],
                 [-1, 1, 1, 1, 1, 1, 1, -1, -1],
                 [-1, -1, 1, 1, 1, 1, 1, -1, -1],
                 [-1, -1, -1, -1, -1, -1, -1, -1, -1]])
