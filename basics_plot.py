import matplotlib.pyplot as plt
import numpy as np

### default matplotlib colours:
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def simple_plot(fig, ax, xaxis, yaxis, xlabel, ylabel, figsz = (12, 16), fontsz = 12, markr = 'o',
                ylog = False, xlog = False, reduce = False, save = False, namefig = None, title = None, alpha = 1):
    if reduce == True:
        xaxis = xaxis[0:-1:10]
        yaxis = yaxis[0:-1:10]

    
    ax.plot(xaxis, yaxis, markr, ms = 3, alpha = alpha)
    ax.set_xlabel(xlabel, fontsize = fontsz)
    ax.set_ylabel(ylabel, fontsize = fontsz)

    if title != None:
        ax.set_title(title, fontsize = fontsz)

    if save == True:
        fig.savefig(namefig)
    return fig, ax