import Environment,GUI,RL
import Utils
import glob
import numpy as np
import math

# Define the cars
# For training only one car is used(First one). For run_only all cars are simulated.
# For a change in dynamics of the car: initial state -> position and orientation(radians), car size, limits, sensor range and sensor angle(randians), make sure the algorithm is trained first.
cars = [
    { 'id':1, 'state':[1,0.3,0], 'L':0.3, 'W':0.1, 'v_limit':2 , 'gamma_limit':0.61, 'sensors':[{'range':2.0,'angle':0.53},{'range':2.0,'angle':0},{'range':2.0,'angle':-0.53}] },
    { 'id':2, 'state':[0.4,0.3,1], 'L':0.3, 'W':0.1, 'v_limit':2 , 'gamma_limit':0.61, 'sensors':[{'range':2.0,'angle':0.53},{'range':2.0,'angle':0},{'range':2.0,'angle':-0.53}] },
    { 'id':3, 'state':[3,0.3,-0.4], 'L':0.3, 'W':0.1, 'v_limit':2 , 'gamma_limit':0.61, 'sensors':[{'range':2.0,'angle':0.53},{'range':2.0,'angle':0},{'range':2.0,'angle':-0.53}] },
    { 'id':4, 'state':[1.5,0.4,-0.1], 'L':0.3, 'W':0.1, 'v_limit':2 , 'gamma_limit':0.61, 'sensors':[{'range':2.0,'angle':0.53},{'range':2.0,'angle':0},{'range':2.0,'angle':-0.53}] },
    { 'id':5, 'state':[2,0.3,0.3], 'L':0.3, 'W':0.1, 'v_limit':2 , 'gamma_limit':0.61, 'sensors':[{'range':2.0,'angle':0.53},{'range':2.0,'angle':0},{'range':2.0,'angle':-0.53}] },
    { 'id':6, 'state':[1,0.5,0.7], 'L':0.3, 'W':0.1, 'v_limit':2 , 'gamma_limit':0.61, 'sensors':[{'range':2.0,'angle':0.53},{'range':2.0,'angle':0},{'range':2.0,'angle':-0.53}] },
]

env_definition = {
    'BIGBOX':[(0,0),(40,0),(40,40),(0,40)],
    'BOX':[(0,0),(10,0),(10,10),(0,10)],
    'dest':[5,5],
    'dest_radius':0.5
}

# Dynamics and control update rate
dt = 0.1

# Fixed set of velocity and steering set at regular interval.
# Change the GOALS to desired update of velocity and steering
def static_control(env_select):
    Environment.env_generator(env_definition,env_select=env_select)
    env = Environment.Environment(env_definition,10000)
    gui = GUI.GUI(env_definition,cars,['Average loss','Total reward','Running reward'],trace=True)
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
def user_control(env_select):
    Environment.env_generator(env_definition,env_select=env_select)
    env = Environment.Environment(env_definition,10000)
    gui = GUI.GUI(env_definition,cars,['Average loss','Total reward','Running reward'],trace=True)
    car_objects = [Environment.Car(c) for c in cars]
    env.compute_interaction(car_objects)
    d = {'w':[0.5,0.0],'s':[-0.5,0.0],'a':[0.5,0.6],'d':[0.5,-0.6],'i':[1.5,0.0],'k':[-1.5,0.0],'j':[1.5,0.1],'l':[1.5,-0.1]}
    while(True):
        user_ip = raw_input()
        if user_ip == 'q': break
        [v,s] = d[user_ip] if (user_ip in d) else [0,0]
        for agent in car_objects:
            agent.set_velocity(v)
            agent.set_steering(s)
        for i in range(10):
            debug_data = ''
            for i in range(len(car_objects)):
                if car_objects[i].state == 'collided':
                    debug_data += 'Car '+str(i)+'\nCollided!\n\n'
                    continue
                elif car_objects[i].state == 'destination':
                    debug_data += 'Car '+str(i)+'\nDestination!\n\n'
                    continue
                car_objects[i].update(dt)
                s_r = car_objects[i].get_sensor_reading()
                gui.update(i,car_objects[i].get_state())
                delta = car_objects[i].get_state_to_train(env.max_delta)
                #debug_data += 'Car '+str(i)+'\nSensor readings:'+', '.join(['{:.2f}'.format(x) for x in s_r])+'\nCar score='+'{:.2f}'.format(car_objects[i].score)+'\n'
                debug_data += 'Car '+str(i)+'\nSensor readings:'+', '.join(['{:.2f}'.format(x) for x in s_r])+'\nCar score='+', '.join(['{:.2f}'.format(y) for y in delta])+'\n'
            env.compute_interaction(car_objects)
            gui.update_debug_info(debug_data)
            gui.refresh()

def reinfrocement_neural_network_control(env_select,load_weights=None,run_only=False,random_seed=None,config_file='config.ini'):
    run=run_only
    rl_prams = Utils.configurator(config_file)
    Environment.env_generator(env_definition,env_select=env_select)
    env = Environment.Environment(env_definition,rl_prams['max_steps'])
    gui = GUI.GUI(env_definition,cars,['Average loss','Total reward','Running reward'],trace=True)
    car_objects = [Environment.Car(c) for c in cars]
    env.compute_interaction(car_objects)
    rl = RL.QLearning_NN(rl_prams, run_only=run, sample_state=car_objects[0].get_state_to_train(10), n_agents=len(car_objects))
    if load_weights is not None:
        if load_weights=='all':
            run=True
        else:
            rl.load_weights(load_weights)
    if random_seed is not None: rl.random_seed(random_seed)

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

    def change_destination():
        if gui.mouse_click_loaction[0] is not None:
            env.change_destination(float(gui.mouse_click_loaction[0]),float(gui.mouse_click_loaction[1]))
        if gui.destination[0]!=env.destination.x:
            gui.init_destination((env.destination.x,env.destination.y),gui.env['dest_radius'],reinit=True)
            gui.mouse_click_loaction = [None,None]

    initialize(run_state=run)
    while(1):
        new_run_state = check_run_button(current_state=run)
        if new_run_state is not None: run=new_run_state
        change_destination()
        if run==True:
            terminals,terminal_states = rl.run_step(car_objects,env,dt)
            for t,ts in zip(terminals,terminal_states):
                print 'Car',t,':',ts
            for i in range(len(car_objects)): gui.update(i,car_objects[i].get_state())
            env.compute_interaction(car_objects)
            gui.refresh()
        else:
            terminals,terminal_states,debug,log = rl.learn_step(car_objects,env,dt)
            if len(terminals)>0:
                if debug is not None:
                    gui.update_debug_info(debug)
                    log = log[0]
                    gui.update_graph(log['epoch'],log['avg_loss'],gui.graphs[0])
                    gui.update_graph(log['epoch'],log['total_reward'],gui.graphs[1])
                    gui.update_graph(log['epoch'],log['running_reward'],gui.graphs[2])
                for t,ts in zip(terminals,terminal_states):
                    gui.update(t,ts,draw_car=False,force_end_line=True)
                gui.refresh()
            show_cars = (car_objects[0].epoch%100==0)
            for i in range(len(car_objects)):
                gui.update(i,car_objects[i].get_state(),draw_car=show_cars)
            if show_cars==True: gui.refresh()

if __name__=='__main__':
    args = Utils.parse_args()
    if args.control=='static':
        static_control(env_select=args.env)
    elif args.control=='user':
        user_control(env_select=args.env)
    elif args.control=='rl':
        reinfrocement_neural_network_control(env_select=args.env,load_weights=args.load_weights,run_only=args.run_only,random_seed=args.random_seed,config_file='config.ini')
