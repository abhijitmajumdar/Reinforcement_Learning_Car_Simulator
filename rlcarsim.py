from Simulator import Environment,GUI,RL,Utils

# User controls the movement of all cars simultaneously, using 'w','a','s','d' for forward, left, reverse and right.
# Use 'q' to quit and 'r' to reset and randomize agents
# Can be used to sense how the system works(sensor readings, collisions and car score)
def user_control(config_file,arena_select=None,continuous_control=False):
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    if arena_select is None: arena_select=env_definition['arena_select'] # Override config arena select if specified in command line arguments
    cars = [Environment.Car(car) for car in car_definitions]
    env = Environment.Environment(env_definition,arena_select=arena_select)
    gui = GUI.GUI(env_definition,arena_select,car_definitions,['Average loss','Total reward','Running reward'],trace=True)
    env.check_agent_connections(*cars)
    env.randomize(rl_params['random_agent_position'],rl_params['random_destination_position'],*cars)
    env.compute_interaction(*cars) # Necessary to ensure vaild values
    gui.init_destination(False,*cars)
    # Controls for the user, change as needed
    d = {'w':[0.5,0.0],'s':[-0.5,0.0],'a':[0.5,0.6],'d':[0.5,-0.6]} if continuous_control==False else {'w':[0.02,0.0],'s':[-0.02,0.0],'a':[0.0,0.04],'d':[0.0,-0.04]}
    loop_for = 1 if continuous_control==True else 10
    instruction_string = 'User commands\n'+'\n'.join([str(key)+': '+str(d[key]) for key in d])+'\n'
    instruction_string += 'r: reset\nq: quit\n\n'

    def change_destination():
        gui.init_destination(True,*cars)
        if gui.mouse_click_loaction[0] is not None:
            for car in cars:
                env.change_destination(car,float(gui.mouse_click_loaction[0]),float(gui.mouse_click_loaction[1]))
            gui.mouse_click_loaction = [None,None]

    while(True):
        for i in range(loop_for):
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
        user_ip = gui.get_userinput()
        if user_ip == 'q': break
        if user_ip == 'r':
            for idx,car in enumerate(cars):
                gui.update(idx,car.get_state(),draw_car=False,force_end_line=True)
                car.reset()
            env.randomize(True,True,*cars)
        [v,s] = d[user_ip] if (user_ip in d) else [0,0]
        if continuous_control==True:
            for car in cars:
                car.increment_velocity(v)
                car.increment_steering(s)
        else:
            for car in cars:
                car.set_velocity(v)
                car.set_steering(s)

def rl_control_dqn(config_file,arena_select=None,load_weights=None,testing=False):
    run='test' if testing is True else 'learn'
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    if arena_select is None: arena_select=env_definition['arena_select'] # Override config arena select if specified in command line arguments
    cars = [Environment.Car(car) for car in car_definitions]
    car = cars[0]
    env = Environment.Environment(env_definition,arena_select=arena_select)
    gui = GUI.GUI(env_definition,arena_select,car_definitions,env_definition['graphs'],trace=True)
    env.check_agent_connections(car)
    env.compute_interaction(car) # Necessary to ensure vaild values
    gui.init_destination(False,car)
    rl = RL.DQN(rl_params, testing=testing, sample_state=car.get_partial_state(),load_weights=load_weights)

    def initialize(run_state):
        car.reset()
        env.compute_interaction(car)
        car.get_sensor_reading()
        if run_state=='test':
            env.randomize(rl_params['random_agent_position'],rl_params['random_destination_position'],car)
            env.set_max_steps(2*env_definition['max_steps'])
            gui.enable_trace(remove_traces=True)
            gui.set_run_select(gui.runs[1])
            gui.update_debug_info('[Testing]\n'+'Currently learned weights loaded')
        else:
            env.randomize(rl_params['random_agent_position'],rl_params['random_destination_position'],car)
            env.set_max_steps(env_definition['max_steps'])
            gui.enable_trace(remove_traces=True)
            gui.set_run_select(gui.runs[0])
            gui.update_debug_info('[Training]\n')
        env.compute_interaction(car)
        rl.init_state_buffer(env,env_definition['dt'],car) # Necessary beacuse the simulator computes agent history, even when its disabled(when the history is set to 1)

    def check_run_button(current_state):
        if gui.get_run_select()==gui.runs[0] and current_state=='test':
            print '\n\n\nLearning\n'
            initialize(run_state='learn')
            return 'learn'
        elif gui.get_run_select()==gui.runs[1] and current_state=='learn':
            print '\n\n\nTesting\n'
            initialize(run_state='test')
            return 'test'
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
        if gui.get_userinput()=='q': break
        if run=='test':
            terminals,terminal_states,physical_states = rl.run_step(env,env_definition['dt'],car,True)
            for i,term in enumerate(terminals):
                gui.update(term,terminal_states[i],draw_car=False,force_end_line=True)
                print 'Car',i,':',physical_states[i]
            gui.update(0,car.get_state())
            gui.refresh()
        else:
            terminals,terminal_states,physical_states,debug,log = rl.learn_step(env,env_definition['dt'],car)
            if debug is not None:
                gui.update_debug_info(debug)
                gui.update_graph(log['epoch'],log['avg_loss'],env_definition['graphs'][0])
                gui.update_graph(log['epoch'],log['total_reward'],env_definition['graphs'][1])
                gui.update_graph(log['epoch'],log['running_reward'],env_definition['graphs'][2])
            for i,term in enumerate(terminals):
                gui.update(term,terminal_states[i],draw_car=False,force_end_line=True)
            show_car = (car.epoch%100==0)
            gui.update(0,car.get_state(),draw_car=show_car)
            if show_car==True or len(terminals)>0: gui.refresh()

def rl_control_mvedql(config_file,arena_select=None,load_weights=None,testing=False):
    run='test' if testing is True else 'learn'
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    if arena_select is None: arena_select=env_definition['arena_select'] # Override config arena select if specified in command line arguments
    cars = [Environment.Car(car) for car in car_definitions]
    env = Environment.Environment(env_definition,arena_select=arena_select)
    gui = GUI.GUI(env_definition,arena_select,car_definitions,env_definition['graphs'],trace=True)
    env.check_agent_connections(*cars)
    env.compute_interaction(*cars) # Necessary to ensure vaild values
    gui.init_destination(False,*cars)
    rl = RL.MVEDQL(rl_params, testing=testing, sample_state=cars[0].get_partial_state(),load_weights=load_weights)

    def initialize(run_state):
        for car in cars: car.reset()
        env.compute_interaction(*cars)
        for car in cars: car.get_sensor_reading()
        if run_state=='test':
            env.randomize(rl_params['random_agent_position'],rl_params['random_destination_position'],*cars)
            env.set_max_steps(2*env_definition['max_steps'])
            gui.enable_trace(remove_traces=True)
            gui.set_run_select(gui.runs[1])
            gui.update_debug_info('[Testing]\n'+'Currently learned weights loaded')
        else:
            env.randomize(rl_params['random_agent_position'],rl_params['random_destination_position'],*cars)
            env.set_max_steps(env_definition['max_steps'])
            gui.enable_trace(remove_traces=True)
            gui.set_run_select(gui.runs[0])
            gui.update_debug_info('[Training]\n')
        env.compute_interaction(*cars)
        rl.init_state_buffer(env,env_definition['dt'],None,*cars) # Necessary beacuse the simulator computes agent history, even when its disabled(when the history is set to 1)

    def check_run_button(current_state):
        if gui.get_run_select()==gui.runs[0] and current_state=='test':
            print '\n\n\nLearning\n'
            initialize(run_state='learn')
            return 'learn'
        elif gui.get_run_select()==gui.runs[1] and current_state=='learn':
            print '\n\n\nTesting\n'
            initialize(run_state='test')
            return 'test'
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
        if gui.get_userinput()=='q': break
        if run=='test':
            terminals,terminal_states,physical_states = rl.run_step(env,env_definition['dt'],True,*cars)
            for i,term in enumerate(terminals):
                gui.update(term,terminal_states[i],draw_car=False,force_end_line=True)
                print 'Car',i,':',physical_states[i]
            for i in range(len(cars)): gui.update(i,cars[i].get_state())
            gui.refresh()
        else:
            terminals,terminal_states,physical_states,debug,log = rl.learn_step(env,env_definition['dt'],*cars)
            if debug is not None:
                gui.update_debug_info(debug)
                gui.update_graph(log['epoch'],log['avg_loss'],env_definition['graphs'][0])
                gui.update_graph(log['epoch'],log['total_reward'],env_definition['graphs'][1])
                gui.update_graph(log['epoch'],log['running_reward'],env_definition['graphs'][2])
            for i,term in enumerate(terminals):
                gui.update(term,terminal_states[i],draw_car=False,force_end_line=True)
            show_car = (cars[0].epoch%100==0)
            for i in range(len(cars)): gui.update(i,cars[i].get_state(),draw_car=show_car)
            if show_car==True or len(terminals)>0: gui.refresh()

def checkpoint_run(config_file,arena_select=None,load_weights=None):
    if load_weights is None:
        raise Exception('To run checkpoint, weights need to be sepcified using load_weights')
    dests = [(24,2),(27,5),(16.5,1.5),(18,8.7),(16.4,4.4),(19,4),(26.5,5),(19,5.2)]
    rl_params,car_definitions,env_definition = Utils.configurator(config_file)
    if arena_select is None: arena_select=env_definition['arena_select'] # Override config arena select if specified in command line arguments
    cars = [Environment.Car(car) for car in car_definitions]
    car = cars[0]
    env = Environment.Environment(env_definition,arena_select=arena_select)
    gui = GUI.GUI(env_definition,arena_select,car_definitions,env_definition['graphs'],trace=True)
    env.check_agent_connections(car)
    env.compute_interaction(car) # Necessary to ensure vaild values
    gui.init_destination(False,car)
    rl = RL.DQN(rl_params, testing=True, sample_state=car.get_partial_state(),load_weights=load_weights)

    # Initialize
    env.set_max_steps(2*env_definition['max_steps'])
    gui.enable_trace(remove_traces=True)
    rl.init_state_buffer(env,env_definition['dt'],car) # Necessary beacuse the simulator computes car history, even when its disabled(when the history is set to 1)
    for idx,pt in enumerate(dests):
        gui.create_marker(pt,'x',0.15)
        gui.create_label(pt,str(idx+1))
    d_idx = 0
    car.set_destination(dests[d_idx])
    gui.create_marker((car.x,car.y),'o',0.1)
    gui.create_marker((car.x,car.y),'arrow',0.5,car.omega)

    while(d_idx<len(dests)):
        if gui.get_userinput()=='q': break
        terminals,terminal_states,physical_states = rl.run_step(env,env_definition['dt'],car,reset=False)
        if car.physical_state=='collided' or car.physical_state=='destination' or car.physical_state=='timeup':
            if car.physical_state=='collided':
                gui.update(0,car.get_state(),draw_car=False,force_end_line=True)
                continue
            d_idx += 1
            if d_idx>=len(dests): break
            car.set_destination(dests[d_idx])
            car.physical_state = 'running'
            env.compute_interaction(car)
            gui.sleep(2)
        gui.init_destination(True,car)
        gui.update(0,car.get_state())
        gui.refresh()

if __name__=='__main__':
    args = Utils.parse_args()
    if args.control=='user':
        user_control(config_file=args.config,arena_select=args.arena,continuous_control=args.cts)
    elif args.control=='dqn':
        rl_control_dqn(config_file=args.config,arena_select=args.arena,load_weights=args.load_weights,testing=args.test)
    elif args.control=='mvedql':
        rl_control_mvedql(config_file=args.config,arena_select=args.arena,load_weights=args.load_weights,testing=args.test)
    elif args.control=='checkpoint':
        checkpoint_run(config_file=args.config,arena_select=args.arena,load_weights=args.load_weights)
