import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from IPython import display
from utils import letter_A, letter_B

class HopfieldNet():
    def __init__(self, n_steps=10, thr=0.0):
        self.n_steps = n_steps
        self.thr_val = thr
        self.update = np.vectorize(lambda x: -1 if x < 0 else +1)
        self.state = None

    def train(self, data):
        n, s = data.shape
        self.n_units = s
        self.W = np.zeros([s, s])

        if isinstance(self.thr_val, (float, int)):
            self.thr = self.thr_val * np.ones(s)
        elif isinstance(self.thr_val, np.ndarray):
            self.thr = self.thr_val
        else:
            raise ValueError('Treshold should be provided as int/float ' + \
                             'or numpy.ndarray')

        if n / s > 0.138:
            raise ValueError('Number of patterns exceeded the memory limit.')

        for pattern in data:
            self.W = self.W + np.outer(pattern, pattern)

        np.testing.assert_array_equal(self.W, data.T.dot(data))

        self.W[np.diag_indices(s)] = 0
        # self.W = self.W / self.n_units
        # test

    def recall(self, input, n_steps=None, synchronous=False,
               store_energy=False,
               warm_start=False):

        if not warm_start or self.state is None:
            self.state = input.copy()

        if n_steps is None:
            n_steps = self.n_steps

        if not hasattr(self, 'E') or store_energy is False:
            self.E = []

        if synchronous:
            for i in range(n_steps):
                self.state = np.where(np.dot(self.W, self.state) > self.thr, 1,
                                      -1)
                self.E.append(self.energy(self.state))

        if not synchronous:
            for i in range(n_steps):
                sel = np.random.randint(low=0, high=self.n_units)
                act = np.dot(self.W[sel, :], self.state)
                if act > self.thr[sel]:
                    self.state[sel] = 1
                else:
                    self.state[sel] = -1
                self.E.append(self.energy(self.state))
        return self.state

    def energy(self, state):
        return -0.5 * np.sum(self.W * np.outer(state, state))

    def plot_weights(self):
        f, ax = plt.subplots(1, 1, figsize=[6, 6])
        sns.heatmap(data=self.W, ax=ax, square=True)
        return f



ne_widget = widgets.BoundedIntText(
    value=20,
    min=1, 
    max=100, 
    description='Number of updates:',
    disabled=False)    

p_widget = widgets.FloatSlider(
    value=0.1,
    min=0,
    max=0.5,
    step=0.05,
    description='Amount of noise:')


sel_input = widgets.Dropdown(
    options=['A', 'B'],
    value='A',
    description='Input:',
    disabled=False)


def corrupt_data(input_data, p_noise=0.1):
    noise = np.random.choice([1, -1], size=input_data.shape, p=[1-p_noise, p_noise])
    return input_data * noise


class simulation_0():
    def __init__(self, train_data, input_data=None):

        self.train_data = train_data
        if input_data is not None:
            self.input_data = input_data
        self.side = np.sqrt(self.train_data.shape[1]).astype(int)

    def plot(self, state):

        f, ax = plt.subplots(2, 2, figsize=[15, 10])

        # plot the pattern to be retrieved
        if self.train_data.shape[0] == 1:
            sns.heatmap(self.train_data.reshape(self.side, self.side), ax=ax[0, 0], vmin=-1,
                        vmax=1, cmap='RdBu_r')

            # plot the current state
        try:
            sns.heatmap(state.reshape(self.side, self.side), ax=ax[0, 1],
                        vmin=-1, vmax=1, cmap='RdBu_r')
        except:
            pass

        # plot the weights
        sns.heatmap(self.net.W, ax=ax[1, 0], xticklabels=False,
                    yticklabels=False, cmap='RdBu_r')

        # plot the energy
        try:
            ax[1, 1].scatter(range(len(self.net.E)), self.net.E)
            ax[1, 1].plot(range(len(self.net.E)), self.net.E)
            ax[1, 1].set_xticks(
                range(0, len(self.net.E), len(self.net.E) // 10))
            ax[1, 1].set_xticklabels(
                range(0, len(self.net.E), len(self.net.E) // 10))

        except (ValueError, AttributeError):
            pass

        # name stuff
        ax[0, 0].set_title('Pattern to retrieve')
        ax[0, 1].set_title('Network state')
        ax[1, 0].set_title('Network weights')
        ax[1, 1].set_title('Energy')
        ax[1, 1].set_xlabel('Iteration')

        # display
        # If the buttons keep appearing comment the line below
        # self.display_buttons()
        display.display(f)
        display.clear_output(wait=True)



    def initialize_and_train_button(self, b):
        # initialize the net
        self.net = HopfieldNet()
        self.net.train(self.train_data)

        # plotting
        self.plot(state=self.input_data)

    def update_button(self, b):
        # update the state
        state = self.net.recall(self.input_data,
                                n_steps=ne_widget.value,
                                synchronous=False,
                                store_energy=True,
                                warm_start=True)

        # plotting
        self.plot(state=state)


    def start_simulation(self):
        self.display_buttons()

        
    def display_buttons(self):
        display.display(ne_widget)
            
        button = widgets.Button(description="Initialize and train")
        display.display(button)
        button.on_click(self.initialize_and_train_button)

        button = widgets.Button(description="Update network")
        display.display(button)
        button.on_click(self.update_button)

        
class simulation():
    
    def __init__(self, train_data, input_data=None):
        
        self.train_data = train_data
        if input_data is not None:
            self.input_data = input_data
        self.side = np.sqrt(self.train_data.shape[1]).astype(int)

    def plot(self, state):
        display.clear_output(wait=True)

        f, ax = plt.subplots(2, 2, figsize=[15, 10])
        
        # plot the pattern to be retrieved
        if sel_input.value == 'A':
            sns.heatmap(letter_A, ax=ax[0, 0], vmin=-1, vmax=1, cmap='RdBu_r') 
        if sel_input.value == 'B':
            sns.heatmap(letter_B, ax=ax[0, 0], vmin=-1, vmax=1, cmap='RdBu_r') 
        
        # plot the current state 
        try:
            sns.heatmap(state.reshape(self.side, self.side), ax=ax[0, 1], vmin=-1, vmax=1, 
                       cmap='RdBu_r')
        except:
            pass
        
        # plot the weights
        sns.heatmap(self.net.W, ax=ax[1, 0], xticklabels=False, yticklabels=False, 
                    cmap='RdBu_r')
        
        # plot the energy
        try:
            ax[1, 1].scatter(range(len(self.net.E)), self.net.E)
            ax[1, 1].plot(range(len(self.net.E)), self.net.E)
            ax[1, 1].set_xticks(range(0, len(self.net.E), len(self.net.E)//10))
            ax[1, 1].set_xticklabels(range(0, len(self.net.E), len(self.net.E)//10))
            
        except (ValueError, AttributeError):
            pass
        
        # name stuff
        ax[0, 0].set_title('Pattern to retrieve')
        ax[0, 1].set_title('Network state')
        ax[1, 0].set_title('Network weights')
        ax[1, 1].set_title('Energy')
        ax[1, 1].set_xlabel('Iteration')
       
        # If the buttons keep appearing comment the line below
        # self.display_buttons()
        display.display(f)
        display.clear_output(wait=True)


    def initialize_and_train_button(self, b):
        # initialize the net
        self.net = HopfieldNet()
        self.net.train(self.train_data)
        
        # plotting
        self.plot(state=None)

        
    def update_button(self, b):
        # update the state
        state = self.net.recall(self.input_data, 
                                n_steps=ne_widget.value, 
                                synchronous=False, 
                                store_energy=True, 
                                warm_start=True)
        
        # plotting
        self.plot(state=state)

    
    def generate_random_input(self, b):
        self.net = HopfieldNet()
        self.net.train(self.train_data)
        # generate noisy input
        if sel_input.value == 'A':
            input_data = letter_A.reshape(81)
        if sel_input.value == 'B':
            input_data = letter_B.reshape(81)

        self.input_data = corrupt_data(input_data, p_noise=p_widget.value) 
        self.net.E = []   
        self.plot(state=self.input_data)


    def start_simulation(self):
        self.display_buttons()
        
    
    def display_buttons(self):
        display.display(sel_input)
        display.display(p_widget)
        display.display(ne_widget)

        button = widgets.Button(description="Initialize and train")
        display.display(button)
        button.on_click(self.initialize_and_train_button)

        button = widgets.Button(description="Generate random input")
        display.display(button)
        button.on_click(self.generate_random_input)

        button = widgets.Button(description="Update network")
        display.display(button)
        button.on_click(self.update_button)



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
