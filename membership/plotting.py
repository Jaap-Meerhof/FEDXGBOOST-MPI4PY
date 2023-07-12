import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axis as Axis


def prettyPlot(data):
    print(str(data))
    return


def plot_data(data: np.array, labels, destination= 'plot.png', name='Sample Text', suptext="Sample Text"):
    """will plot the different variables against the first column of data (the value tested against)

    Args:
        data (_type_): twodimentional array of variables
        labels (_type_): corresponding labels
    """
    amount_of_plots = len(data[0])
    nrows = 3
    ncols = (amount_of_plots//3) + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,20))
    plt.subplots_adjust(hspace=0.4)

    x = data[:, 0]
    y = data[:, 1:]
    y_all = [list(tmp) for tmp in zip(*y)]

    for i, y in enumerate(y_all):
        ax= axs[i//ncols, i%ncols]
        ax.plot(x, y)
        ax.grid(True)
        ax.set_xticks(x)
        ax.set_yticks(y)
        ax.set_xlabel(labels[0])
        ax.set_xticklabels(x, rotation=45, ha='right')
        ax.set_ylabel(labels[i+1])
        # for i in range(len(x)):
        #     ax.text(x[i], y[i], f'({x[i]}, {y[i]})', ha='center', va='bottom')
        ax.set_title(labels[i+1])

    ax = axs[(i+1)//ncols, (i+1)%ncols]
    ax.text(0.5, 0.5, name, 
        horizontalalignment='center', verticalalignment='center',
        fontsize=10, color='black')
    ax.set_xticks([])
    ax.set_yticks([])


    if nrows*ncols > amount_of_plots:
        for i in range(amount_of_plots, nrows, ncols):
            fig.delaxes[i//ncols, i%ncols]
    plt.tight_layout()
    fig.suptitle(suptext)
    # fig.text(ncols, nrows, 'testtesttest')
    plt.savefig("results/" + destination)
    # plt.title("test2")
    # plt.show()
    pass