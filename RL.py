import numpy as np
import random
import tensorflow as tf
import keras

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
    def __init__(self,rl_params,weights_save_dir,run_only,sample_state):
        # Dont use all of my GPU memory
        keras.backend.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True))))
        self.parameters = dict(rl_params)
        self.weights_save_dir = weights_save_dir
        self.parameters['output_length'] = len(self.parameters['actions'])
        self.parameters['state_dimension'] = len(sample_state)
        self.epoch = 0
        self.replay = ReplayMemory(length=self.parameters['buffer_length'], state_length=self.parameters['state_dimension'], defined_terminal_states=['collided','timeup','destination'], minimum_buffer_length=self.parameters['replay_start_at']) if run_only==False else None
        self.itr,self.avg_loss,self.avg_score = 0,0,0
        self.train_hist = None
        self.total_reward = 0
        self.log = {'avg_loss':[],'total_reward':[],'state':[],'running_reward':[],'epoch':[]}

    def random_seed(self,seed):
        random.seed(seed)

    def generate_nn(self):
        self.model = keras.models.Sequential()
        weights_init = keras.initializers.Constant(value=0.001) #'lecun_uniform'
        activation = None
        self.model.add(keras.layers.Dense(20, kernel_initializer=weights_init, input_shape=(self.parameters['state_dimension'],), activation=activation))
        #self.model.add(keras.layers.LeakyReLU(alpha=self.parameters['leak_alpha']))
        self.model.add(keras.layers.Dense(12, kernel_initializer=weights_init, activation=activation))
        #self.model.add(keras.layers.LeakyReLU(alpha=self.parameters['leak_alpha']))
        self.model.add(keras.layers.Dense(self.parameters['output_length'], kernel_initializer=weights_init, activation='linear'))
        #self.model.add(keras.layers.LeakyReLU(alpha=self.parameters['leak_alpha']))
        # I found Adam is more stable(than SGD/RMSprop) in handling new samples of (X,y) and overfitting, it does still oscillate but in a more subtle manner
        optim = keras.optimizers.Adam(lr=self.parameters['lr_alpha'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #lr_alpha=0.001
        self.model.compile(optimizer=optim, loss='mse')

    def load_weights(self,weights):
        self.model.load_weights(weights)

    def take_action(self,agent,dt,epsilon_override=None):
        dstate = agent.get_state_to_train()
        q_vals = self.model.predict(dstate.reshape(1,self.parameters['state_dimension']),batch_size=1,verbose=0)
        if epsilon_override is not None:
            epsilon = epsilon_override
        else:
            epsilon = self.parameters['epsilon']
        if (random.random() > epsilon):
            action = random.randint(0,self.parameters['output_length']-1)
        else:
            action = (np.argmax(q_vals))
        v,s = self.parameters['actions'][action]
        agent.set_velocity(v)
        agent.set_steering(s)
        agent.update(dt)
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

    def check_terminal_state_and_log(self,agent,env,reward):
        self.itr += 1
        self.avg_loss = 0 if self.train_hist is None else (self.avg_loss+self.train_hist.history['loss'][0])
        self.total_reward += reward
        terminal_state,debug_data = None,None
        if agent.state=='collided' or agent.state=='destination' or agent.state=='timeup':
            self.epoch += 1
            if self.replay.minimum_buffer_filled()==True and self.parameters['epsilon']<self.parameters['max_epsilon']:
                self.parameters['epsilon'] += self.parameters['epsilon_step']
            self.avg_loss /= self.itr
            self.log['avg_loss'].append(self.avg_loss)
            self.log['total_reward'].append(self.total_reward)
            self.log['state'].append(agent.state)
            self.avg_score = self.total_reward if self.epoch==1 else (0.99*self.avg_score + 0.01*self.total_reward)
            self.log['running_reward'].append(self.avg_score)
            self.log['epoch'].append(self.epoch)
            if self.epoch%5==0:
                np.save('./log',self.log)
                self.model.save_weights(self.weights_save_dir+'rlcar_epoch_'+str(self.epoch).zfill(5))
                print 'Epoch ',self.epoch,'Epsilon=',self.parameters['epsilon'],'Run=',agent.state,'Avg score=',self.avg_score,'Avg loss=',self.avg_loss
                debug_data = '[Training]\n'+'Epoch '+str(self.epoch)+'\nEpsilon='+str(self.parameters['epsilon'])+'\nRun='+str(agent.state)+'\nAvg score='+'{:.2f}'.format(self.avg_score)+'\nAvg loss='+str(self.avg_loss)
            self.avg_loss,self.itr,self.total_reward = 0,0,0
            terminal_state = agent.get_state()
            agent.reset()
            if self.parameters['random_car_position']==True:
                agent.random_state([5,5,0],4,np.pi)
                #env.randomize()
            env.compute_interaction([agent])
        return terminal_state,debug_data,self.log

    def check_terminal_state(self,agent,env):
        terminal_state = None
        if agent.state=='collided' or agent.state=='destination' or agent.state=='timeup':
            terminal_state = agent.state
            agent.reset()
            if self.parameters['random_car_position']==True:
                agent.random_state([5,5,0],4,np.pi)
            env.compute_interaction([agent])
        return terminal_state

    def learn_step(self,agent,env,dt):
        dstate,action_taken = self.take_action(agent,dt)
        env.compute_interaction([agent])
        new_dstate = agent.get_state_to_train()
        reward = self.reward_function(agent)
        self.replay.add(dstate,action_taken,reward,new_dstate,agent.state)
        self.train_nn()
        return self.check_terminal_state_and_log(agent,env,reward)

    def run_step(self,agent,env,dt):
        self.take_action(agent,dt,epsilon_override=1.0)
        return self.check_terminal_state(agent,env)
