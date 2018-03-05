import os
import configobj
import argparse
import shutil

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
    config = configobj.ConfigObj(config_file,unrepr=True)
    rl_p,cars,env_definition = {},[],{}
    for key in config['Reinforcement Parameters']:
        rl_p[key] = config['Reinforcement Parameters'][key]
    for key in config['Process triggers']:
        rl_p[key] = config['Process triggers'][key]
    rl_p['actions'] = [config['Actions'][y] for y in config['Actions']]
    rl_p['logdir'] = config['Log']['logdir']
    rl_p['activation'] = config['Network']['activation']
    rl_p['layers'] = config['Network']['layers']
    for key in config['Cars']:
        car = {}
        for subkey in config['Cars'][key]:
            car[subkey] = config['Cars'][key][subkey]
        car['sensors'] = [car['sensors'][k] for k in car['sensors']]
        cars.append(car)
    env_definition = dict(config['Environment'])
    return rl_p,cars,env_definition

def log_file(f,logdir):
    shutil.copy(f,logdir)

def parse_args():
    parser = argparse.ArgumentParser(description="RL-Car Project")
    parser.add_argument("--control", help="static/user/rl",default='static')
    parser.add_argument("--run_only", dest='run_only', action='store_true', help="epsilon=1,no_training")
    parser.add_argument("--load_weights", help="path to load saved weights")
    parser.add_argument("--env", help="BOX/BIGBOX",default='BOX')
    parser.add_argument("--random_seed", help="Run reproducable results", default=None, type=int)
    parser.add_argument("--config", help="path to config file",default='config.ini')
    return parser.parse_args()
