import argparse

import matplotlib.pyplot as plt

import torch

def plot_loss_map(train_losses, val_maps):
    plt.ion()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.grid(True)

    ax1.plot(train_losses, color='red')
    ax2.plot(val_maps)

    plt.ioff()
    max_window()
    plt.show()

# insert this before plt.show()
def max_window():
    backend = plt.get_backend()

    if backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        ### works on Ubuntu??? >> did NOT working on windows
        # mng.resize(*mng.window.maxsize())
        mng.window.state('zoomed') #works fine on Windows!
    elif backend == 'wxAgg':
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif backend == 'Qt4Agg' or backend == 'Qt5Agg':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='Pneumonia Mask RCNN Plotter')
    parser.add_argument('--state_file', default='./states.pth', help='state file path')
    flags = parser.parse_args()

    # read states
    states = torch.load(flags.state_file)

    # plot
    plot_loss_map(states['loss_history'], states['val_map_history'])
