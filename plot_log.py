import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import glob
import seaborn as sns
import pandas as pd

COLOURS = ['b-','r-','g-','b','r','c']

def load_logfile(f):
    return np.load(f).reshape((1,))[0]

def init_plot(n):
    f, plts = plt.subplots(3, n)
    if n==1: plts = plts.reshape((-1,n))
    plts[0][0].set_ylabel('Running reward')
    plts[1][0].set_ylabel('Average loss')
    plts[2][0].set_ylabel('Total reward')
    axes,plots = [],[]
    for i in range(len(plts[0])):
        p1, = plts[0,i].plot([],[],COLOURS[i],linewidth=3,antialiased=False)
        p2, = plts[1,i].plot([],[],COLOURS[i],linewidth=3,antialiased=False)
        p3, = plts[2,i].plot([],[],COLOURS[i],linewidth=3,antialiased=False)
        plots.append((p1,p2,p3))
        axes.append((plts[0,i],plts[1,i],plts[2,i]))
        plts[2,i].set_xlabel('Epochs')
    return plots,f,axes

def plot_log(l,plts,axes):
    plts[0].set_data(l['epoch'],l['running_reward'])
    axes[0].relim()
    axes[0].autoscale_view()
    plts[1].set_data(l['epoch'],l['avg_loss'])
    axes[1].relim()
    axes[1].autoscale_view()
    plts[2].set_data(l['epoch'],l['total_reward'])
    axes[2].relim()
    axes[2].autoscale_view()

def plot_rewards(logs, plot_points=False, truncate=0, labels=None):
    epoch,reward,cats = [],[],[]
    ls = [str(n) for n in range(len(logs))] if ((labels is None) or (len(labels)!=len(logs))) else labels
    for n,l in enumerate(logs):
        for i,x in enumerate(l['epoch']):
            if truncate!=0 and i>truncate: continue
            epoch.append(x)
            reward.append(l['total_reward'][i])
            cats.append(ls[n])
    categories = pd.Categorical(cats, categories=ls)
    df = pd.DataFrame({'epoch':epoch,'total_reward':reward,'log':categories})
    ax = sns.lmplot(x='epoch', y='total_reward', data=df, hue='log', truncate=True,order=6,fit_reg=True,scatter=plot_points,ci=95,legend_out=False)
    ax.set_axis_labels("Episodes", "Average Reward")

def compare_rewards(logdir,ns,truncate,show_points=False,labels=None):
    files = [sorted(glob.glob(logdir+str(i)+'/log*'))[0] for i in ns]
    logs = [load_logfile(f) for f in files]
    plot_rewards(logs,show_points,truncate,labels)
    plt.show()

def show_log(logdir,loop,truncate,show_points=False):
    files = sorted([name for name in glob.glob(logdir+'log*')])
    plots,fig,axes = init_plot(len(files))
    while(True):
        logs = [load_logfile(f) for f in files]
        for i in range(len(logs)):
            plot_log(logs[i],plots[i],axes[i])
        if loop==False:
            plot_rewards(logs,show_points,truncate)
            plt.pause(1)
            plt.show()
            break
        else:
            plt.pause(10)

def parse_args():
    parser = argparse.ArgumentParser(description="RL-Car Project")
    parser.add_argument("--logdir", help="path to load log")
    parser.add_argument("--loop", dest='loop', action='store_true', help="update at regular intervals")
    parser.add_argument("--compare", type=str, help="plot of reward", default=None)
    parser.add_argument("--points", dest='points', action='store_true', help="show the points")
    parser.add_argument("--truncate", type=int, default=0)
    parser.add_argument("--labels", nargs='+', type=str, default=None)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    if args.compare is not None:
        idx=[]
        compare = args.compare.split(',')
        for entry in compare:
            if '-' in entry:
                b,e = [int(x) for x in entry.split('-')]
                idx += list(range(b,e+1))
            else:
                idx.append(int(entry))
        compare_rewards(logdir=args.logdir,ns=idx,truncate=args.truncate,show_points=args.points,labels=args.labels)
    else:
        show_log(logdir=args.logdir,loop=args.loop,truncate=args.truncate,show_points=args.points)
