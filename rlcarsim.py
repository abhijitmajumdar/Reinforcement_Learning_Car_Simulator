from Simulator import Environment,GUI,RL,Utils

# User controls the movement of all cars simultaneously, using 'w','a','s','d' for forward, left, reverse and right.
# Use 'q' to quit and 'r' to reset and randomize agents
# Can be used to sense how the system works(sensor readings, collisions and car score)
def user_control(env_select,config_file,multi_agent=False):
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    cars = [Environment.Car(car) for car in car_definitions] if multi_agent==True else [Environment.Car(car_definitions[0])]
    env = Environment.Environment(env_definition,env_select=env_select)
    gui = GUI.GUI(env_definition,env_select,car_definitions,['Average loss','Total reward','Running reward'],trace=True)
    env.compute_interaction(*cars) # Necessary to ensure vaild values
    gui.init_destination(False,*cars)
    # Controls for the user, change as needed
    d = {'w':[0.5,0.0],'s':[-0.5,0.0],'a':[0.5,0.6],'d':[0.5,-0.6],'i':[1.5,0.0],'k':[-1.5,0.0],'j':[1.5,0.1],'l':[1.5,-0.1]}
    instruction_string = 'User commands\n'+'\n'.join([str(key)+':'+str(d[key]) for key in d])+'\n\n'

    def change_destination():
        gui.init_destination(True,*cars)
        if gui.mouse_click_loaction[0] is not None:
            for car in cars:
                env.change_destination(car,float(gui.mouse_click_loaction[0]),float(gui.mouse_click_loaction[1]))
            gui.mouse_click_loaction = [None,None]

    while(True):
        user_ip = raw_input()
        if user_ip == 'q': break
        if user_ip == 'r':
            for idx,car in enumerate(cars):
                gui.update(idx,car.get_state(),draw_car=False,force_end_line=True)
                car.reset()
            env.randomize(True,True,*cars)
        [v,s] = d[user_ip] if (user_ip in d) else [0,0]
        for car in cars:
            car.set_velocity(v)
            car.set_steering(s)
        for i in range(10):
            debug_data = ''
            for idx,car in enumerate(cars):
                if car.physical_state == 'collided' or car.physical_state == 'destination':
                    debug_data += 'Car '+str(idx)+'\n'+car.physical_state+'!\n\n'
                    continue
                car.update(env_definition['dt'])
                s_r = car.get_sensor_reading()
                gui.update(idx,car.get_state())
                delta = car.get_partial_state()
                debug_data += 'Car '+str(idx)+'\nSensor readings:'+', '.join(['{:.2f}'.format(x) for x in s_r])+'\nPartial state='+', '.join(['{:.2f}'.format(y) for y in delta])+'\n'
            env.compute_interaction(*cars)
            gui.update_debug_info(instruction_string+debug_data)
            change_destination()
            gui.refresh()

def reinfrocement_neural_network_control(env_select,config_file,load_weights=None,run_only=False):
    run=run_only
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    car = Environment.Car(car_definitions[0])
    env = Environment.Environment(env_definition,env_select=env_select)
    gui = GUI.GUI(env_definition,env_select,[car_definitions[0]],env_definition['graphs'],trace=True)
    env.compute_interaction(car) # Necessary since rl initialization expects a valid state
    gui.init_destination(False,car)
    rl = RL.DQN(rl_params, run_only=run, sample_state=car.get_partial_state(),load_weights=load_weights)

    def initialize(run_state):
        car.reset()
        env.compute_interaction(car)
        car.get_sensor_reading()
        if run_state==True:
            env.set_max_steps(1500)
            gui.disable_trace(remove_traces=True)
            gui.set_run_select(gui.runs[1])
            gui.update_debug_info('[Testing]\n'+'Currently learned weights loaded')
        else:
            env.randomize(rl_params['random_agent_position'],rl_params['random_destination_position'],car)
            env.set_max_steps(env_definition['max_steps'])
            gui.enable_trace()
            gui.set_run_select(gui.runs[0])
            gui.update_debug_info('[Training]\n')
        env.compute_interaction(car)

    def check_run_button(current_state):
        if gui.get_run_select()==gui.runs[0] and current_state==True:
            print '\n\n\nLearning\n'
            initialize(run_state=False)
            return False
        elif gui.get_run_select()==gui.runs[1] and current_state==False:
            print '\n\n\nRun only\n'
            initialize(run_state=True)
            return True
        else:
            return current_state

    def change_destination():
        gui.init_destination(True,car)
        if gui.mouse_click_loaction[0] is not None:
            env.change_destination(car,float(gui.mouse_click_loaction[0]),float(gui.mouse_click_loaction[1]))
            gui.mouse_click_loaction = [None,None]

    initialize(run_state=run)
    while(1):
        run = check_run_button(current_state=run)
        change_destination()
        if run==True:
            terminal_state,physical_state = rl.run_step(car,env,env_definition['dt'])
            if physical_state is not None:
                print 'Car',':',physical_state
                gui.update(0,terminal_state,draw_car=False,force_end_line=True)
            gui.update(0,car.get_state())
            env.compute_interaction(car)
            gui.refresh()
        else:
            terminal_state,physical_state,debug,log = rl.learn_step(car,env,env_definition['dt'])
            if debug is not None:
                gui.update_debug_info(debug)
                gui.update_graph(log['epoch'],log['avg_loss'],env_definition['graphs'][0])
                gui.update_graph(log['epoch'],log['total_reward'],env_definition['graphs'][1])
                gui.update_graph(log['epoch'],log['running_reward'],env_definition['graphs'][2])
            if physical_state is not None:
                gui.update(0,terminal_state,draw_car=False,force_end_line=True)
                gui.refresh()
            show_car = (car.epoch%100==0)
            gui.update(0,car.get_state(),draw_car=show_car)
            if show_car==True: gui.refresh()

if __name__=='__main__':
    args = Utils.parse_args()
    if args.control=='user':
        user_control(env_select=args.env,config_file=args.config)
    elif args.control=='multi':
        user_control(env_select=args.env,config_file=args.config,multi_agent=True)
    elif args.control=='dqn':
        reinfrocement_neural_network_control(env_select=args.env,config_file=args.config,load_weights=args.load_weights,run_only=args.run_only)
