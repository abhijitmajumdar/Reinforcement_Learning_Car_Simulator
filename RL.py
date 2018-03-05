import numpy as np
import random
import tensorflow as tf
import keras
import Utils

class ReplayMemory():
    def __init__(self,length,state_length,defined_terminal_states,minimum_buffer_length=None):
        # state,action,reward,new_state,agent_state
        self.idx = 0
        self.length = length
        self.buffer_full = False
        self.old_states = np.empty((length,state_length),dtype=float)
        self.old_actions = np.empty(length,dtype=int)
        self.rewards = np.empty(length,dtype=float)
        self.new_states = np.empty((length,state_length),dtype=float)
        self.terminal_state = np.empty(length,dtype=bool)
        self.defined_terminal_states = defined_terminal_states
        self.minimum_buffer_length = minimum_buffer_length

    def add(self,state,action,reward,new_state,phy_state):
        self.old_states[self.idx,:] = state
        self.old_actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.new_states[self.idx,:] = new_state
        self.terminal_state[self.idx] = True if phy_state in self.defined_terminal_states else False
        self.idx += 1
        if self.idx>=self.length:
            self.idx=0
            self.buffer_full = True
            print '!!!!!!!!!!!!!!!Buffer filled!!!!!!!!!!!!!!!!!!!'

    def sample(self,size):
        rand_idxs = []
        if self.buffer_full==False:
            check_size = size
            if self.minimum_buffer_length is not None:
                check_size = self.minimum_buffer_length
            if self.idx<check_size:
                return None,None,None,None,None
            rand_idxs = np.arange(self.idx)
        else:
            rand_idxs = np.arange(self.length)
        rand_idxs = random.sample(rand_idxs,size)
        return self.old_states[rand_idxs],self.old_actions[rand_idxs],self.rewards[rand_idxs],self.new_states[rand_idxs],self.terminal_state[rand_idxs]

    def minimum_buffer_filled(self):
        return True if self.buffer_full is True else (True if self.minimum_buffer_length is None else (True if self.idx>self.minimum_buffer_length else False))

class QLearning_NN():
    def __init__(self,rl_params,run_only,sample_state,n_agents=None):
        # Dont use all of my GPU memory
        keras.backend.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True))))
        self.parameters = dict(rl_params)
        if run_only==False: self.parameters['logdir'] = Utils.check_directory(self.parameters['logdir'])
        self.parameters['output_length'] = len(self.parameters['actions'])
        self.parameters['state_dimension'] = len(sample_state)
        self.parameters['n_agents'] = n_agents
        self.epoch,self.total_reward,self.avg_score = np.zeros(self.parameters['n_agents']),np.zeros(self.parameters['n_agents']),np.zeros(self.parameters['n_agents'])
        self.itr,self.avg_loss,self.train_hist = 0,0,None
        self.log = [{'avg_loss':[],'total_reward':[],'state':[],'running_reward':[],'epoch':[]} for i in range(self.parameters['n_agents'])]
        self.replay = ReplayMemory(length=int(self.parameters['buffer_length']), state_length=self.parameters['state_dimension'], defined_terminal_states=['collided','timeup','destination'], minimum_buffer_length=self.parameters['replay_start_at']) if run_only==False else None
        self.avg_score_differential = 0
        self.variance = 0
        self.generate_nn()
        keras.utils.plot_model(self.model,to_file=self.parameters['logdir']+'model.png',show_shapes=True, show_layer_names=True, rankdir='LR')
        np.save(self.parameters['logdir']+'parameters',self.parameters)

    def random_seed(self,seed):
        random.seed(seed)

    def generate_nn(self):
        self.model = keras.models.Sequential()
        weights_init = keras.initializers.he_normal(5) # or keras.initializers.he_uniform(5)
        self.model.add(keras.layers.Dense(self.parameters['layers'][0], kernel_initializer=weights_init, input_shape=(self.parameters['state_dimension'],), activation=self.parameters['activation']))
        for i in range(1,len(self.parameters['layers'])):
            self.model.add(keras.layers.Dense(self.parameters['layers'][i], kernel_initializer=weights_init, activation=self.parameters['activation']))
            #self.model.add(keras.layers.LeakyReLU(alpha=self.parameters['leak_alpha']))
        self.model.add(keras.layers.Dense(self.parameters['output_length'], kernel_initializer=weights_init, activation='linear'))
        optim = keras.optimizers.Adam(lr=self.parameters['lr_alpha'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #lr_alpha=0.001
        self.model.compile(optimizer=optim, loss='mse')

    def load_weights(self,weights):
        self.model.load_weights(weights)

    def take_action(self,agents,env,dt,epsilon_override=None,variance=0.5):
        dstate = np.array([agent.get_state_to_train(env.max_delta) for agent in agents])
        q_vals = self.model.predict(dstate,batch_size=len(agents),verbose=0)
        if epsilon_override is not None:
            epsilon = [epsilon_override]*len(agents)
        elif self.parameters['adaptive_multi_epsilon_policy']==True:
            epsilon = np.concatenate(([self.parameters['epsilon']],np.random.uniform(low=max(0,self.parameters['epsilon']-variance),high=min(1,self.parameters['epsilon']+variance),size=len(agents)-1)))
        else:
            epsilon = [self.parameters['epsilon']]*len(agents)
        r = random.random()
        action = [np.argmax(q_vals[i]) if r<epsilon[i] else np.random.random_integers(0,self.parameters['output_length']-1) for i in range(len(agents))]
        for i in range(len(agents)):
            v,s = self.parameters['actions'][action[i]]
            agents[i].set_velocity(v)
            agents[i].set_steering(s)
            agents[i].update(dt)
        return dstate,action

    def reward_function(self,agent):
        if agent.state == 'timeup':
            reward = self.parameters['timeup_reward']
        elif agent.state == 'collided':
            reward = self.parameters['collision_reward']
        elif agent.state == 'destination':
            reward = self.parameters['destination_reward']
        else:
            reward = 0
            #reward = agent.score # Encourages the car to MOVE, not necessarily forward, infact moving in circles is encouraged
            #reward = -30+agent.score # Encourages the car to crash and end its misery
            #reward = -1 # To factor in time but encourages the car to crash and end its misery. Useful if destination reward is high
            #reward = 1 if agent.score-agent.prev_score<0.05 else (-1 if agent.score-agent.prev_score>0.05 else 0)
        return reward

    def train_nn(self):
        old_states,old_actions,rewards,new_states,car_state = self.replay.sample(self.parameters['minibatchsize'])
        if old_states is None:
            return
        old_qvals = self.model.predict(old_states, batch_size=self.parameters['minibatchsize'])
        new_qvals = self.model.predict(new_states, batch_size=self.parameters['minibatchsize'])
        maxQs = np.max(new_qvals, axis=1)
        y_train = old_qvals
        non_terminal_idx = np.where(car_state == False)[0]
        term_idx = np.where(car_state == True)[0]
        y_train[non_terminal_idx, old_actions[non_terminal_idx].astype(int)] = rewards[non_terminal_idx] + (self.parameters['gamma'] * maxQs[non_terminal_idx])
        y_train[term_idx, old_actions[term_idx].astype(int)] = rewards[term_idx]
        X_train = old_states
        self.train_hist = self.model.fit(X_train, y_train, batch_size=self.parameters['batchsize'], epochs=1, verbose=0)

    def check_terminal_state_and_log(self,agents,env,reward):
        self.itr += 1
        self.avg_loss = 0 if self.train_hist is None else (self.avg_loss+self.train_hist.history['loss'][0])
        terminals,terminal_states,debug_data = [],[],None
        for i,agent in enumerate(agents):
            agent.total_reward += reward[i]
            if agent.state=='collided' or agent.state=='destination' or agent.state=='timeup':
                agent.epoch += 1
                self.log[i]['total_reward'].append(agent.total_reward)
                self.log[i]['state'].append(agent.state)
                #self.log[i]['running_reward'].append(agent.total_reward if agent.epoch==1 else (0.99*self.log[i]['running_reward'][-1] + 0.01*agent.total_reward))
                #self.log[i]['running_reward'].append(np.mean(self.log[i]['total_reward'][-100:]))
                self.log[i]['running_reward'].append(np.mean(self.log[i]['total_reward']) if len(self.log[i]['total_reward'])>10 else min(self.log[i]['total_reward']))
                self.log[i]['epoch'].append(agent.epoch)
                if i==0:
                    self.log[i]['avg_loss'].append(self.avg_loss/self.itr)
                    if self.replay.minimum_buffer_filled()==True and self.parameters['epsilon']<self.parameters['max_epsilon']:
                        self.parameters['epsilon'] += self.parameters['epsilon_step']
                    self.avg_loss,self.itr = 0,0
                    if agent.epoch%20==0:
                        self.model.save_weights(self.parameters['logdir']+'rlcar_epoch_'+str(agent.epoch).zfill(5))
                        print 'Epoch ',agent.epoch,'Epsilon=',self.parameters['epsilon'],'Run=',agent.state,'Avg score=',self.log[i]['running_reward'][-1],'Avg loss=',self.log[i]['avg_loss'][-1]
                        debug_data = '[Training]\n'+'Epoch '+str(agent.epoch)+'\nEpsilon='+str(self.parameters['epsilon'])+'\nRun='+str(agent.state)+'\nAvg score='+'{:.2f}'.format(self.log[i]['running_reward'][-1])+'\nAvg loss='+str(self.log[i]['avg_loss'][-1])
                        self.variance = 1-Utils.noraml_scale(self.log[i]['running_reward'][-1],min(self.log[i]['running_reward']),max(self.log[i]['running_reward'])) if len(self.log[i]['running_reward'])>40 else min(self.log[i]['running_reward'])
                        self.variance = max(0.1,min(0.7,self.variance))
                else:
                    self.log[i]['avg_loss'].append(0)
                if agent.epoch%5==0: np.save(self.parameters['logdir']+'log_A'+str(i),self.log[i])
                terminal_states.append(agent.get_state())
                terminals.append(i)
                agent.reset()
                if self.parameters['random_agent_position']==True:
                    agent.random_state([5,5,0],4,np.pi)
                if self.parameters['random_destination_position']==True:
                    if self.parameters['different_destinations']==True:
                        env.randomize([agent])
                    elif i==0:
                        env.randomize(agents)

                env.compute_interaction([agent])
        return terminals,terminal_states,debug_data,self.log

    def check_terminal_state(self,agents,env):
        terminals,terminal_states = [],[]
        for i,agent in enumerate(agents):
            if agent.state=='collided' or agent.state=='destination' or agent.state=='timeup':
                terminal_states.append(agent.state)
                terminals.append(i)
                agent.reset()
                if self.parameters['random_agent_position']==True:
                    agent.random_state([5,5,0],4,np.pi)
                env.compute_interaction([agent])
        return terminals,terminal_states

    def learn_step(self,agents,env,dt):
        dstate,action_taken = self.take_action(agents,env,dt,variance=self.variance)
        env.compute_interaction(agents)
        new_dstate = [agent.get_state_to_train(env.max_delta) for agent in agents]
        reward = [self.reward_function(agent) for agent in agents]
        agent_state = [agent.state for agent in agents]
        for i in range(len(agents)):
            self.replay.add(dstate[i],action_taken[i],reward[i],new_dstate[i],agent_state[i])
        self.train_nn()
        return self.check_terminal_state_and_log(agents,env,reward)

    def run_step(self,agents,env,dt):
        self.take_action(agents,env,dt,epsilon_override=1.0)
        return self.check_terminal_state(agents,env)
