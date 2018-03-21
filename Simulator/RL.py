import numpy as np
import random
import tensorflow as tf
import keras

def noraml_scale(val,mi,ma):
    if mi==ma:
        return mi
    return (float(val)-mi)/(ma-mi)

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


class QLearning_NN(object):
    def __init__(self,rl_params,run_only,sample_state,load_weights=None):
        # Dont use all of my GPU memory
        keras.backend.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True))))
        self.parameters = dict(rl_params)
        self.parameters['output_length'] = len(self.parameters['actions'])
        self.parameters['state_dimension'] = len(sample_state)
        # Neural Network
        self.generate_q_value_approximator()
        # Initialize parameters
        if self.parameters['random_seed'] is not None: self.random_seed(self.parameters['random_seed'])
        if load_weights is not None: self.load_weights(load_weights)
        # Replay Memory
        self.replay = ReplayMemory(length=self.parameters['buffer_length'], state_length=self.parameters['state_dimension'], defined_terminal_states=[key for key in self.parameters['terminal_state_rewards']], minimum_buffer_length=self.parameters['replay_start_at']) if run_only==False else None
        # Local log
        self.itr,self.avg_loss,self.train_hist = 0,0,None
        self.epoch,self.total_reward,self.avg_score = 0,0,0
        # Log
        self.log = {'avg_loss':[],'total_reward':[],'state':[],'running_reward':[],'epoch':[]}
        keras.utils.plot_model(self.model,to_file=self.parameters['logdir']+'model.png',show_shapes=True, show_layer_names=True, rankdir='LR')
        np.save(self.parameters['logdir']+'parameters',self.parameters)

    def random_seed(self,seed):
        random.seed(seed)

    def generate_nn(self):
        self.optimizer = keras.optimizers.Adam(lr=self.parameters['lr_alpha'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #lr_alpha=0.001
        self.optimizer_loss = 'mse'
        self.weights_init = keras.initializers.he_normal(5) # or keras.initializers.he_uniform(5)
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.parameters['layers'][0], kernel_initializer=self.weights_init, input_shape=(self.parameters['state_dimension'],), activation=self.parameters['activation']))
        for i in range(1,len(self.parameters['layers'])):
            self.model.add(keras.layers.Dense(self.parameters['layers'][i], kernel_initializer=self.weights_init, activation=self.parameters['activation']))
            #self.model.add(keras.layers.LeakyReLU(alpha=self.parameters['leak_alpha']))
        self.model.add(keras.layers.Dense(self.parameters['output_length'], kernel_initializer=self.weights_init, activation='linear'))
        self.model.compile(optimizer=self.optimizer, loss=self.optimizer_loss)

    def load_weights(self,weights):
        self.model.load_weights(weights)

    def generate_q_value_approximator(self):
        self.generate_nn()

    def reward_function(self,agent):
        if agent.physical_state in self.parameters['terminal_state_rewards']:
            reward = self.parameters['terminal_state_rewards'][agent.physical_state]
        else:
            reward = self.parameters['normal_reward']
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

class DQN(QLearning_NN):
    def clone_network(self,src_model):
        return keras.models.Sequential.from_config(src_model.get_config())

    def copy_weights(self,src_model,dest_model):
        dest_model.set_weights(src_model.get_weights())

    def generate_target_q_network(self):
        self.model_target = self.clone_network(self.model)
        self.model_target.compile(optimizer=self.optimizer, loss=self.optimizer_loss)
        self.copy_weights(self.model,self.model_target)
        self.target_update_counter = 0

    def generate_q_value_approximator(self):
        self.generate_nn()
        self.generate_target_q_network()

    def load_weights(self,weights):
        self.model.load_weights(weights)
        self.model_target.load_weights(weights)

    def take_action(self,agent,dt,epsilon_override=None):
        dstate = agent.get_partial_state().reshape((1,-1))
        q_vals = self.model.predict(dstate,batch_size=1,verbose=0)
        if epsilon_override is not None:
            epsilon = epsilon_override
        else:
            epsilon = self.parameters['epsilon']
        r = random.random()
        action = np.argmax(q_vals) if r<epsilon else np.random.random_integers(0,self.parameters['output_length']-1)
        v,s = self.parameters['actions'][action]
        agent.set_velocity(v)
        agent.set_steering(s)
        agent.update(dt)
        return dstate,action

    def train_nn(self):
        old_states,old_actions,rewards,new_states,car_state = self.replay.sample(self.parameters['minibatchsize'])
        if old_states is None:
            return
        old_qvals = self.model.predict(old_states, batch_size=self.parameters['minibatchsize'])
        new_qvals = self.model_target.predict(new_states, batch_size=self.parameters['minibatchsize'])
        maxQs = np.max(new_qvals, axis=1)
        y_train = old_qvals
        non_terminal_idx = np.where(car_state == False)[0]
        term_idx = np.where(car_state == True)[0]
        y_train[non_terminal_idx, old_actions[non_terminal_idx].astype(int)] = rewards[non_terminal_idx] + (self.parameters['gamma'] * maxQs[non_terminal_idx])
        y_train[term_idx, old_actions[term_idx].astype(int)] = rewards[term_idx]
        X_train = old_states
        self.train_hist = self.model.fit(X_train, y_train, batch_size=self.parameters['batchsize'], epochs=1, verbose=0)
        self.target_update_counter += 1
        if self.target_update_counter>=self.parameters['target_network_update_frequency']:
            self.copy_weights(self.model,self.model_target)
            self.target_update_counter = 0

    def check_terminal_state_and_log(self,agent,env,reward):
        self.itr += 1
        self.avg_loss = 0 if self.train_hist is None else (self.avg_loss+self.train_hist.history['loss'][0])
        terminal_state,physical_state,debug_data = None,None,None
        agent.total_reward += reward
        if agent.physical_state=='collided' or agent.physical_state=='destination' or agent.physical_state=='timeup':
            agent.epoch += 1
            self.log['total_reward'].append(agent.total_reward)
            self.log['state'].append(agent.physical_state)
            #self.log['running_reward'].append(np.mean(self.log['total_reward'][-100:]))
            self.log['running_reward'].append(np.mean(self.log['total_reward']) if len(self.log['total_reward'])>10 else min(self.log['total_reward']))
            self.log['epoch'].append(agent.epoch)
            self.log['avg_loss'].append(self.avg_loss/self.itr)
            if self.replay.minimum_buffer_filled()==True and self.parameters['epsilon']<self.parameters['max_epsilon']:
                self.parameters['epsilon'] += self.parameters['epsilon_step']
            self.avg_loss,self.itr = 0,0
            if agent.epoch%self.parameters['save_interval']==0:
                self.model.save_weights(self.parameters['logdir']+'rlcar_epoch_'+str(agent.epoch).zfill(5))
                print 'Epoch ',agent.epoch,'Epsilon=',self.parameters['epsilon'],'Run=',agent.physical_state,'Avg score=',self.log['running_reward'][-1],'Avg loss=',self.log['avg_loss'][-1]
                debug_data = '[Training]\n'+'Epoch '+str(agent.epoch)+'\nEpsilon='+str(self.parameters['epsilon'])+'\nRun='+str(agent.physical_state)+'\nAvg score='+'{:.2f}'.format(self.log['running_reward'][-1])+'\nAvg loss='+str(self.log['avg_loss'][-1])
                np.save(self.parameters['logdir']+'log_A'+str(0),self.log)
            terminal_state = agent.get_state()
            physical_state = agent.physical_state
            agent.reset()
            env.randomize(self.parameters['random_agent_position'],self.parameters['random_destination_position'],agent)
            env.compute_interaction(agent)
        return terminal_state,physical_state,debug_data,self.log

    def check_terminal_state(self,agent,env):
        terminal_state,physical_state = None,None
        if agent.physical_state=='collided' or agent.physical_state=='destination' or agent.physical_state=='timeup':
            terminal_state = agent.get_state()
            physical_state = agent.physical_state
            agent.reset()
            env.randomize(self.parameters['random_agent_position'],self.parameters['random_destination_position'],agent)
            env.compute_interaction(agent)
        return terminal_state,physical_state

    def learn_step(self,agent,env,dt):
        dstate,action_taken = self.take_action(agent,dt)
        env.compute_interaction(agent)
        new_dstate = agent.get_partial_state()
        reward = self.reward_function(agent)
        agent_physical_state = agent.physical_state
        self.replay.add(dstate,action_taken,reward,new_dstate,agent_physical_state)
        self.train_nn()
        return self.check_terminal_state_and_log(agent,env,reward)

    def run_step(self,agent,env,dt):
        self.take_action(agent,dt,epsilon_override=1.0)
        return self.check_terminal_state(agent,env)
