import os
import configparser
import argparse

def check_directory(directory):
    i = 0
    while(True):
        i += 1
        d = directory+str(i)+'/'
        if not os.path.exists(d):
            os.makedirs(d)
            break
    return directory+str(i)+'/'

def noraml_scale(val,mi,ma):
    if mi==ma:
        return mi
    return (float(val)-mi)/(ma-mi)

def configurator(config_file):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)
    rl_p = {}
    for key in config['Reinforcement Parameters']:
        rl_p[key] = float(config['Reinforcement Parameters'][key])
    rl_p['minibatchsize'] = int(rl_p['minibatchsize'])
    rl_p['batchsize'] = int(rl_p['batchsize'])
    for key in config['Process triggers']:
        rl_p[key] = config['Process triggers'].getboolean(key)
    rl_p['actions'] = [[float(x) for x in config['Actions'][y].split(',')] for y in config['Actions']]
    rl_p['logdir'] = config['Log']['logdir']
    rl_p['activation'] = config['Network']['activation']
    rl_p['layers'] = [int(x) for x in config['Network']['layers'].split(',')]
    return rl_p

def parse_args():
    parser = argparse.ArgumentParser(description="RL-Car Project")
    parser.add_argument("--control", help="static/user/rl",default='static')
    parser.add_argument("--run_only", dest='run_only', action='store_true', help="epsilon=1,no_training")
    parser.add_argument("--load_weights", help="path to load saved weights, or \'all\' to load all available weights in succession")
    parser.add_argument("--env", help="BOX/BIGBOX",default='BOX')
    parser.add_argument("--random_seed", help="Run reproducable results", default=None, type=int)
    return parser.parse_args()
