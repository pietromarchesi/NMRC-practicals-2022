import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

mpl.rcParams['figure.dpi'] = 80


def plot_single_neuron_output(w, p, b, tf_name):
    transfer_function = getattr(utils, tf_name)
    output = transfer_function(w * p - b)

    lim = 4
    xx = np.arange(-lim, lim, 0.01)
    yy = transfer_function(w * xx - b)
    f, ax = plt.subplots()
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.plot(xx, yy)
    ax.set_title('Function: %s; w=%s, p=%s, b=%s'
                 % (transfer_function.__name__, w, p, b))

    ax.axvline(p, c='r', linestyle=':')
    ax.axhline(output, c='r', linestyle=':')
    ax.scatter(p, output, c='r', s=50)
    ax.set_ylim([-1.3, 1.3])
    plt.show()
