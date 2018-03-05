import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import glob

COLOURS = ['b-','r-','g-','b','r','c']

def load_logfile(f):
    return np.load(f).reshape((1,))[0]

def init_plot(n):
    f, plts = plt.subplots(2, n)
    if n==1: plts = plts.reshape((-1,n))
    plts[0][0].set_ylabel('Running reward')
    plts[1][0].set_ylabel('Average loss')
    axes,plots = [],[]
    for i in range(len(plts[0])):
        p1, = plts[0,i].plot([],[],COLOURS[i],linewidth=3,antialiased=False)
        p2, = plts[1,i].plot([],[],COLOURS[i],linewidth=3,antialiased=False)
        plots.append((p1,p2))
        axes.append((plts[0,i],plts[1,i]))
        plts[1,i].set_xlabel('Epochs')
    return plots,f,axes

def plot_log(l,plts,axes):
    plts[0].set_data(l['epoch'],l['running_reward'])
    axes[0].relim()
    axes[0].autoscale_view()
    plts[1].set_data(l['epoch'],l['avg_loss'])
    axes[1].relim()
    axes[1].autoscale_view()

def show_log(logdir,loop):
    files = sorted([name for name in glob.glob(logdir+'log*')])
    plots,fig,axes = init_plot(len(files))
    while(True):
        logs = [load_logfile(f) for f in files]
        for i in range(len(logs)):
            plot_log(logs[i],plots[i],axes[i])
        if loop==False:
            plt.pause(1)
            plt.show()
            break
        else:
            plt.pause(10)

def parse_args():
    parser = argparse.ArgumentParser(description="RL-Car Project")
    parser.add_argument("--logdir", help="path to load log")
    parser.add_argument("--loop", dest='loop', action='store_true', help="update at regular intervals")
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    show_log(logdir=args.logdir,loop=args.loop)
