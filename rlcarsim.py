from Simulator import Environment,GUI,RL,Utils

# User controls the movement of all cars simultaneously, using 'w','a','s','d' for forward, left, reverse and right.
# Use 'q' to quit and 'r' to reset and randomize agents
# Can be used to sense how the system works(sensor readings, collisions and car score)
def user_control(config_file,arena_select=None):
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    if arena_select is None: arena_select=env_definition['arena_select'] # Override config arena select if specified in command line arguments
    cars = [Environment.Car(car) for car in car_definitions]
    env = Environment.Environment(env_definition,arena_select=arena_select)
    gui = GUI.GUI(env_definition,arena_select,car_definitions,['Average loss','Total reward','Running reward'],trace=True)
    env.check_agent_connections(*cars)
    env.randomize(True,True,*cars)
    env.compute_interaction(*cars) # Necessary to ensure vaild values
    gui.init_destination(False,*cars)
    # Controls for the user, change as needed
    d = {'w':[0.5,0.0],'s':[-0.5,0.0],'a':[0.5,0.6],'d':[0.5,-0.6]}
    instruction_string = 'User commands\n'+'\n'.join([str(key)+': '+str(d[key]) for key in d])+'\n'
    instruction_string += 'r: reset\nq: quit\n\n'

    def change_destination():
        gui.init_destination(True,*cars)
        if gui.mouse_click_loaction[0] is not None:
            for car in cars:
                env.change_destination(car,float(gui.mouse_click_loaction[0]),float(gui.mouse_click_loaction[1]))
            gui.mouse_click_loaction = [None,None]

    while(True):
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

def reinfrocement_neural_network_control(config_file,arena_select=None,load_weights=None,run_only=False,multi_agent=False):
    run=run_only
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    if arena_select is None: arena_select=env_definition['arena_select'] # Override config arena select if specified in command line arguments
    cars = [Environment.Car(car) for car in car_definitions]
    env = Environment.Environment(env_definition,arena_select=arena_select)
    gui = GUI.GUI(env_definition,arena_select,car_definitions,env_definition['graphs'],trace=True)
    env.check_agent_connections(*cars)
    env.compute_interaction(*cars) # Necessary to ensure vaild values
    gui.init_destination(False,*cars)
    rl_method = RL.DQN
    rl = rl_method(rl_params, run_only=run, sample_state=cars[0].get_partial_state(),load_weights=load_weights)

    def initialize(run_state):
        for car in cars: car.reset()
        env.compute_interaction(*cars)
        for car in cars: car.get_sensor_reading()
        if run_state==True:
            env.set_max_steps(1500)
            gui.disable_trace(remove_traces=True)
            gui.set_run_select(gui.runs[1])
            gui.update_debug_info('[Testing]\n'+'Currently learned weights loaded')
        else:
            env.randomize(rl_params['random_agent_position'],rl_params['random_destination_position'],*cars)
            env.set_max_steps(env_definition['max_steps'])
            gui.enable_trace(remove_traces=True)
            gui.set_run_select(gui.runs[0])
            gui.update_debug_info('[Training]\n')
        env.compute_interaction(*cars)

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
        gui.init_destination(True,*cars)
        if gui.mouse_click_loaction[0] is not None:
            for car in cars:
                env.change_destination(car,float(gui.mouse_click_loaction[0]),float(gui.mouse_click_loaction[1]))
            gui.mouse_click_loaction = [None,None]

    initialize(run_state=run)
    while(1):
        run = check_run_button(current_state=run)
        change_destination()
        if run==True:
            terminals,terminal_states,physical_states = rl.run_step(env,env_definition['dt'],*cars) if multi_agent==True else rl.run_step(env,env_definition['dt'],cars[0])
            for i,term in enumerate(terminals):
                gui.update(i,terminal_states[i],draw_car=False,force_end_line=True)
                print 'Car',i,':',physical_states[i]
            for i in range(len(cars)): gui.update(i,cars[i].get_state())
            env.compute_interaction(*cars)
            gui.refresh()
        else:
            terminals,terminal_states,physical_states,debug,log = rl.learn_step(env,env_definition['dt'],*cars) if multi_agent==True else rl.learn_step(env,env_definition['dt'],cars[0])
            if debug is not None:
                gui.update_debug_info(debug)
                gui.update_graph(log['epoch'],log['avg_loss'],env_definition['graphs'][0])
                gui.update_graph(log['epoch'],log['total_reward'],env_definition['graphs'][1])
                gui.update_graph(log['epoch'],log['running_reward'],env_definition['graphs'][2])
            for i,term in enumerate(terminals):
                gui.update(term,terminal_states[i],draw_car=False,force_end_line=True)
            if len(terminals)>0: gui.refresh()
            show_car = (cars[0].epoch%100==0)
            for i in range(len(cars)): gui.update(i,cars[i].get_state(),draw_car=show_car)
            if show_car==True: gui.refresh()

if __name__=='__main__':
    args = Utils.parse_args()
    if args.control=='user':
        user_control(config_file=args.config,arena_select=args.arena)
    elif args.control=='dqn':
        reinfrocement_neural_network_control(config_file=args.config,arena_select=args.arena,load_weights=args.load_weights,run_only=args.run_only)
