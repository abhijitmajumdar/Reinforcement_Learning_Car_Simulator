import Environment,GUI,RL
import argparse
import glob
import os

# Define the cars
# For training only one car is used(First one). For run_only all cars are simulated.
# For a change in dynamics of the car: initial state -> position and orientation(radians), car size, limits, sensor range and sensor angle(randians), make sure the algorithm is trained first.
cars = [
    { 'id':1, 'state':[1,0.3,0], 'L':0.3, 'W':0.1, 'v':1, 'v_limit':2 ,'gamma':-0.15, 'gamma_limit':0.61, 'sensors':[{'range':1.0,'angle':0.53},{'range':1.0,'angle':0},{'range':1.0,'angle':-0.53}] },
    { 'id':2, 'state':[0.4,0.3,1], 'L':0.3, 'W':0.1, 'v':1, 'v_limit':2 ,'gamma':-0.15, 'gamma_limit':0.61, 'sensors':[{'range':1.0,'angle':0.53},{'range':1.0,'angle':0},{'range':1.0,'angle':-0.53}] },
    { 'id':3, 'state':[3,0.3,-1], 'L':0.3, 'W':0.1, 'v':1, 'v_limit':2 ,'gamma':-0.15, 'gamma_limit':0.61, 'sensors':[{'range':1.0,'angle':0.53},{'range':1.0,'angle':0},{'range':1.0,'angle':-0.53}] },
    { 'id':4, 'state':[1.5,0.4,-3], 'L':0.3, 'W':0.1, 'v':1, 'v_limit':2 ,'gamma':-0.15, 'gamma_limit':0.61, 'sensors':[{'range':1.0,'angle':0.53},{'range':1.0,'angle':0},{'range':1.0,'angle':-0.53}] },
    { 'id':5, 'state':[2,0.3,3], 'L':0.3, 'W':0.1, 'v':1, 'v_limit':2 ,'gamma':-0.15, 'gamma_limit':0.61, 'sensors':[{'range':1.0,'angle':0.53},{'range':1.0,'angle':0},{'range':1.0,'angle':-0.53}] },
    { 'id':6, 'state':[1,0.5,-3.14], 'L':0.3, 'W':0.1, 'v':1, 'v_limit':2 ,'gamma':-0.15, 'gamma_limit':0.61, 'sensors':[{'range':1.0,'angle':0.53},{'range':1.0,'angle':0},{'range':1.0,'angle':-0.53}] },
]

# Different track definitions.
# Make sure that the paths dont overlap.
# Name is irrelevent and can be selected as an argument while running the program.
track = {
    'track_width':0.8,
    'L':[(0,0),(5,0),(7,2),(7,6)],
    'ME':[(0,0),(5,0),(7,2),(7,6),(5,8),(2,8),(0,6),(2,4),(4,4)], # Mirror 'e'
    'S':[(0,0),(5,0),(7,2),(5,4),(2,4),(0,6),(2,8),(7,8)], # 's'
    'SE':[(0,0),(5,0),(7,2),(5,4),(2,4),(0,6),(2,8),(9,8),(9,0)], # 's' extended
    'SS':[(0,0.5),(5,0),(7,2),(5,4),(2,4),(0,6),(2,8),(12,8),(14,6),(12,4),(10,4),(9,2),(10,0),(12,0)], # 's'->mirror 's'
}

# Paramerters used by the reinforcement learning algorithm
rl_parameters = {
    'actions':[(0.5,0),(0.5,0.6),(0.5,-0.6),(1.5,0),(1.5,0.1),(1.5,-0.1)], # Possible actions the agent can take: list of (velocity,steering) tuples
    'epsilon':0.76, # Initial epsilon value to start traingin with
    'max_epsilon':0.98, # Maximum epsilon value to be set as the agent learns
    'epsilon_step':0.02, # Increment to epsilon after every epoch(termination of a run)
    'gamma':0.5, # Discount factor
    'lr_alpha':0.001, # Learning rate for back proportion update of neural netwrok
    'leak_alpha':0.3, # Used by the LeakyReLU activation function after each layer in the NN
    'max_steps':1000, # Timout for the cars to comlete the track, to avoid them going round and round in circles
    'collision_reward':-500, # Reward offered if car collides
    'timeup_reward':-500, # Reward offered if time runs out
    'destination_reward':300, # Reward offered if car completes the track
    'buffer_length':30000, # Size of replay memory to train the NN
    'batchsize':256, # Size of batch to process weight update, varied based on CPU/GPU resources available
    'minibatchsize':256, # Size of batch to update weight at each step. Large values learn more but are slower to process
    'state_dimension':3 #len(cars[0]['sensors']) -> Number of sensors on the 1st car
}

# Dynamics and control update rate
dt = 0.1

# Fixed set of velocity and steering set at regular interval.
# Change the GOALS to desired update of velocity and steering
def static_control(track_select='SS'):
    Environment.track_generator(track,track_select=track_select)
    env = Environment.Environment(track,10000)
    gui = GUI.GUI(track,cars,trace=True)
    car_objects = [Environment.Car(c) for c in cars]
    env.compute_interaction(car_objects)
    GOALS = ((0.33,0),(0.14,0.6),(0.4,0),(0.148,-0.6))
    while(True):
        for goal in GOALS:
            for i in range(len(car_objects)):
                car_objects[i].set_velocity(goal[0])
                car_objects[i].set_steering(goal[1])
            for i in range(100):
                debug_data = ''
                for i in range(len(car_objects)):
                    if car_objects[i].state == 'collided':
                        debug_data += 'Car '+str(i)+'\nCollided!\n\n'
                        continue
                    car_objects[i].update(dt)
                    s_r = car_objects[i].get_sensor_reading()
                    gui.update(i,car_objects[i].get_state())
                    debug_data += 'Car '+str(i)+'\nSensor readings:'+', '.join(['{:.2f}'.format(x) for x in s_r])+'\nCar score='+'{:.2f}'.format(car_objects[i].score)+'\n'
                env.compute_interaction(car_objects)
                gui.update_debug_info(debug_data)
                gui.refresh()

# User controls the movement of all cars simultaneously, using 'w','a','s','d' for forward, left, reverse and right.
# Can be used to sense how the system works(sensor readings, collisions and car score)
def user_control(track_select='SS'):
    Environment.track_generator(track,track_select=track_select)
    env = Environment.Environment(track,10000)
    gui = GUI.GUI(track,cars,trace=True)
    car_objects = [Environment.Car(c) for c in cars]
    env.compute_interaction(car_objects)
    while(True):
        d = {'w':[0.5,0.0],'s':[-0.3,0.0],'a':[0.1,0.6],'d':[0.1,-0.6]}
        try:
            [v,s] = d[raw_input()]
        except:
            [v,s] = [0,0]
        for agent in car_objects:
            agent.set_velocity(v)
            agent.set_steering(s)
        for i in range(10):
            debug_data = ''
            for i in range(len(car_objects)):
                if car_objects[i].state == 'collided':
                    debug_data += 'Car '+str(i)+'\nCollided!\n\n'
                    continue
                car_objects[i].update(dt)
                s_r = car_objects[i].get_sensor_reading()
                gui.update(i,car_objects[i].get_state())
                debug_data += 'Car '+str(i)+'\nSensor readings:'+', '.join(['{:.2f}'.format(x) for x in s_r])+'\nCar score='+'{:.2f}'.format(car_objects[i].score)+'\n'
            env.compute_interaction(car_objects)
            gui.update_debug_info(debug_data)
            gui.refresh()

def reinfrocement_neural_network_control(load_weights=None,run_only=False,track_select='SS',random_seed=None,rl_prams=None):
    run=run_only
    weights_save_dir="./weights/"
    if not os.path.exists(weights_save_dir): os.makedirs(weights_save_dir)
    Environment.track_generator(track,track_select=track_select)
    env = Environment.Environment(track,rl_parameters['max_steps'])
    gui = GUI.GUI(track,cars,trace=True)
    car_objects = [Environment.Car(c) for c in cars]
    rl = RL.QLearning_NN(rl_prams,weights_save_dir=weights_save_dir)
    rl.generate_nn()
    if load_weights is not None:
        if load_weights=='all':
            run=True
        else:
            rl.load_weights(load_weights)
    if random_seed is not None: rl.random_seed(random_seed)
    weight_names = sorted([name for name in glob.glob(weights_save_dir+'*')])
    weight_names_index = 0

    def initialize(run_state):
        env.compute_interaction(car_objects)
        for car in car_objects:
            car.reset()
            car.get_sensor_reading()
        if run_state==True:
            env.set_max_steps(1500)
            gui.remove_traces()
            gui.disable_trace()
            gui.set_run_select(gui.runs[1])
            gui.update_debug_info('[Testing]\n'+'Currently learned weights loaded')
        else:
            env.set_max_steps(rl_prams['max_steps'])
            gui.enable_trace()
            gui.set_run_select(gui.runs[0])
            gui.update_debug_info('[Training]\n')

    def check_run_button(current_state):
        if gui.get_run_select()==gui.runs[0] and current_state==True:
            print '\n\n\nLearning\n'
            initialize(run_state=False)
            return False
        if gui.get_run_select()==gui.runs[1] and run==False:
            print '\n\n\nRun only\n'
            initialize(run_state=True)
            return True
        return None

    initialize(run_state=run)
    while(1):
        new_run_state = check_run_button(current_state=run)
        if new_run_state is not None: run=new_run_state
        if run==True:
            for i,car in enumerate(car_objects):
                terminal = rl.run_step(car,env,dt)
                if terminal is not None:
                    print 'Car',i,':',terminal
                    if i==0:
                        if load_weights=='all' and weight_names_index<len(weight_names):
                            rl.load_weights(weight_names[weight_names_index])
                            gui.update_debug_info('[Testing]\n'+'Weights loaded:\n'+weight_names[weight_names_index])
                            weight_names_index += 1
                gui.update(i,car.get_state())
            env.compute_interaction(car_objects)
            gui.refresh()
        else:
            terminal,debug,epoch,avg_loss,final_score,cross_score = rl.learn_step(car_objects[0],env,dt)
            if terminal is not None:
                if debug is not None:
                    gui.update_debug_info(debug)
                    gui.update_graph(epoch,avg_loss,gui.graphs[0])
                    gui.update_graph(epoch,final_score,gui.graphs[1])
                    gui.update_graph(epoch,cross_score,gui.graphs[2])
                    gui.refresh()
                gui.update(0,terminal,draw_car=False,force_end_line=True)
                gui.refresh()
            if rl.epoch%100==0:
                gui.update(0,car_objects[0].get_state(),draw_car=True)
                gui.refresh()
            else:
                gui.update(0,car_objects[0].get_state(),draw_car=False)

def parse_args():
    parser = argparse.ArgumentParser(description="RL-Car Project")
    parser.add_argument("--control", help="static/user/nn",default='static')
    parser.add_argument("--run_only", dest='run_only', action='store_true', help="epsilon=1,no_training")
    parser.add_argument("--load_weights", help="path to load saved weights, or \'all\' to load all available weights in succession")
    parser.add_argument("--track", help="L/ME/S/SE/SS",default='SS')
    parser.add_argument("--random_seed", help="Run reproducable results", default=None, type=int)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    if args.control=='static':
        static_control(track_select=args.track)
    elif args.control=='user':
        user_control(track_select=args.track)
    elif args.control=='nn':
        reinfrocement_neural_network_control(load_weights=args.load_weights,run_only=args.run_only,track_select=args.track,random_seed=args.random_seed,rl_prams=rl_parameters)
