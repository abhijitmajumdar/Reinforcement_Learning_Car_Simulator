import numpy as np
import random
import keras

class QLearning_NN():
    def __init__(self,rl_params,weights_save_dir):
        self.parameters = dict(rl_params)
        self.weights_save_dir = weights_save_dir
        self.parameters['output_length'] = len(self.parameters['actions'])
        self.epoch = 0
        self.replay,self.replay_index = [],0
        self.itr,self.avg_loss,self.avg_score = 0,0,0
        self.train_hist = None
        self.log = {'avg_loss':[],'final_score':[],'state':[],'cross_score':[],'epoch':[]}

    def random_seed(self,seed):
        random.seed(seed)

    def generate_nn(self):
        self.model = keras.models.Sequential()
        weights_init = keras.initializers.Constant(value=0.1) #'lecun_uniform'
        activation = None
        self.model.add(keras.layers.Dense(20, kernel_initializer=weights_init, input_shape=(self.parameters['state_dimension'],), activation=activation))
        self.model.add(keras.layers.LeakyReLU(alpha=self.parameters['leak_alpha']))
        self.model.add(keras.layers.Dense(self.parameters['output_length'], kernel_initializer=weights_init, activation=activation))
        self.model.add(keras.layers.LeakyReLU(alpha=self.parameters['leak_alpha']))
        # I found Adam is more stable(than SGD/RMSprop) in handling new samples of (X,y) and overfitting, it does still oscillate but in a more subtle manner
        optim = keras.optimizers.Adam(lr=self.parameters['lr_alpha'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #lr_alpha=0.001
        self.model.compile(optimizer=optim, loss='mse')

    def load_weights(self,weights):
        self.model.load_weights(weights)

    def take_action(self,agent,dt,epsilon_override=None):
        sensor_readings = np.array(agent.get_sensor_reading())
        q_vals = self.model.predict(sensor_readings.reshape(1,self.parameters['state_dimension']),batch_size=1,verbose=0)
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
        return sensor_readings,action

    def reward_function(self,agent):
        if agent.state == 'timeup':
            reward = self.parameters['timeup_reward']
        elif agent.state == 'collided':
            reward = self.parameters['collision_reward']
        elif agent.state == 'destination':
            reward = self.parameters['destination_reward']
        else:
            #reward = 0
            #reward = agent.score # Encourages the car to MOVE, not necessarily forward, infact moving in circles is encouraged
            #reward = -30+agent.score # Encourages the car to crash and end its misery
            #reward = -1 # To factor in time but encourages the car to crash and end its misery. Useful if destination reward is high
            reward = 1 if agent.score-agent.prev_score>0 else -1
        return reward

    def train_nn(self,sensor_readings,action,reward,new_sensor_readings,agent_state):
        if (len(self.replay)<self.parameters['buffer_length']):
            self.replay.append((sensor_readings,action,reward,new_sensor_readings,agent_state))
        else:
            self.replay_index += 1
            if self.replay_index>=self.parameters['buffer_length']: self.replay_index=0
            self.replay[self.replay_index] = ((sensor_readings,action,reward,new_sensor_readings,agent_state))
        if (len(self.replay)>self.parameters['replay_start_at']):
            minibatch = random.sample(self.replay, self.parameters['minibatchsize'])
            mb_len = len(minibatch)
            old_states = np.zeros(shape=(mb_len, self.parameters['state_dimension']))
            old_actions = np.zeros(shape=(mb_len,))
            rewards = np.zeros(shape=(mb_len,))
            new_states = np.zeros(shape=(mb_len, self.parameters['state_dimension']))
            car_state = []
            for i, m in enumerate(minibatch):
                old_state_m, action_m, reward_m, new_state_m, car_state_m = m
                old_states[i, :] = old_state_m[...]
                old_actions[i] = action_m
                rewards[i] = reward_m
                new_states[i, :] = new_state_m[...]
                car_state.append(car_state_m)
            car_state = np.array(car_state)
            old_qvals = self.model.predict(old_states, batch_size=mb_len)
            new_qvals = self.model.predict(new_states, batch_size=mb_len)
            maxQs = np.max(new_qvals, axis=1)
            y = old_qvals
            non_term_inds = np.where(car_state == 'running')[0]
            #non_term_inds = np.concatenate((non_term_inds,np.where(car_state == 'destination')[0]))
            term_inds = np.where(car_state == 'timeup')[0]
            term_inds = np.concatenate((term_inds,np.where(car_state == 'collided')[0]))
            term_inds = np.concatenate((term_inds,np.where(car_state == 'destination')[0]))
            y[non_term_inds, old_actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (self.parameters['gamma'] * maxQs[non_term_inds])
            y[term_inds, old_actions[term_inds].astype(int)] = rewards[term_inds]
            X_train = old_states
            y_train = y
            self.train_hist = self.model.fit(X_train, y_train, batch_size=self.parameters['batchsize'], epochs=1, verbose=0)

    def check_terminal_state_and_log(self,agent,env):
        self.itr += 1
        self.avg_loss = 0 if self.train_hist is None else (self.avg_loss+self.train_hist.history['loss'][0])
        terminal_state,debug_data = None,None
        if agent.state=='collided' or agent.state=='destination' or agent.state=='timeup':
            self.epoch += 1
            if (len(self.replay)>=self.parameters['replay_start_at']) and self.parameters['epsilon']<self.parameters['max_epsilon']:
                self.parameters['epsilon'] += self.parameters['epsilon_step']
            self.avg_loss /= self.itr
            self.log['avg_loss'].append(self.avg_loss)
            self.log['final_score'].append(agent.score)
            self.log['state'].append(agent.state)
            if self.avg_loss==0:
                self.log['cross_score'].append(0)
            else:
                self.log['cross_score'].append(agent.score*(1/self.avg_loss))
            self.log['epoch'].append(self.epoch)
            if self.epoch%5==0:
                self.avg_score = sum(self.log['final_score'][self.epoch-5:self.epoch])/5
                np.save('./log',self.log)
                self.model.save_weights(self.weights_save_dir+'rlcar_epoch_'+str(self.epoch).zfill(5))
                print 'Epoch ',self.epoch,'Epsilon=',self.parameters['epsilon'],'Run=',agent.state,'Avg score=',self.avg_score,'Avg loss=',self.avg_loss
                debug_data = '[Training]\n'+'Epoch '+str(self.epoch)+'\nEpsilon='+str(self.parameters['epsilon'])+'\nRun='+str(agent.state)+'\nAvg score='+'{:.2f}'.format(self.avg_score)+'\nAvg loss='+str(self.avg_loss)
            self.avg_loss,self.itr = 0,0
            terminal_state = agent.get_state()
            agent.reset()
            if self.parameters['random_car_position']==True:
                agent.set_state([1+env.route[0].x+(env.track_width*1.2*(random.random()-0.5)),env.route[0].y+(env.track_width*1.2*(random.random()-0.5)),env.start_angle+(random.random()-0.5)])
        return terminal_state,debug_data,self.log['epoch'],self.log['avg_loss'],self.log['final_score'],self.log['cross_score']

    def check_terminal_state(self,agent):
        terminal_state = None
        if agent.state=='collided' or agent.state=='destination' or agent.state=='timeup':
            terminal_state = agent.state
            agent.reset()
        return terminal_state

    def learn_step(self,agent,env,dt):
        sensor_values,action_taken = self.take_action(agent,dt)
        env.compute_interaction([agent])
        new_sensor_values = np.array(agent.get_sensor_reading())
        reward = self.reward_function(agent)
        self.train_nn(sensor_values,action_taken,reward,new_sensor_values,agent.state)
        return self.check_terminal_state_and_log(agent,env)

    def run_step(self,agent,env,dt):
        self.take_action(agent,dt,epsilon_override=1.0)
        return self.check_terminal_state(agent)
