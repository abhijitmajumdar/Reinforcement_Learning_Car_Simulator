#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int16MultiArray,MultiArrayDimension
import message_filters
from Simulator import Environment,GUI,RL,Utils
import numpy as np
import tf

class RLCAR:
    def __init__(self):
        rospy.init_node('rlcarsim', anonymous=False)
        pose_subscriber = message_filters.Subscriber('pose', PoseStamped)
        sensor_subscriber = message_filters.Subscriber('tof', Int16MultiArray)
        self.sense_subscriber = message_filters.TimeSynchronizer([pose_subscriber, sensor_subscriber], 10)
        self.sense_subscriber.registerCallback(self.sense_callback)
        self.motor_publisher = rospy.Publisher('cmd', Int16MultiArray, queue_size=1)
        config_file = rospy.get_param('~config')
        rospy.loginfo('Configuration file:%s',config_file)
        logdir = rospy.get_param('~logdir')
        rospy.loginfo('Weights and logs saved in:%s',logdir)
        load_weights = rospy.get_param('~load_weights',None)
        if not load_weights is None:
            rospy.loginfo('Using pretrained weights:%s',load_weights)
        self.rl_params,car_definitions,self.env_definition = Utils.configurator(config_file)
        self.rl_params['logdir'] = Utils.check_directory(logdir) # overwrite the log directory from node params
        Utils.log_file(config_file,self.rl_params['logdir']) # Save a copy of the config file in the log directory
        arena_select = self.env_definition['arena_select']
        self.car = Environment.Car(car_definitions[0])
        self.env = Environment.Environment(self.env_definition,arena_select=arena_select)
        self.gui = GUI.GUI(self.env_definition,arena_select,car_definitions,self.env_definition['graphs'],trace=True)
        self.env.check_agent_connections(self.car)
        self.env.compute_interaction(self.car) # Necessary to ensure vaild values
        self.gui.init_destination(False,self.car)
        self.rl = RL.DQN(self.rl_params, testing=False, sample_state=self.car.get_partial_state(),load_weights=load_weights)
        self.init_car()
        self.run = self.gui.runs[0] # Learn

    def send_motor_command(self,v,s):
        msg = Int16MultiArray()
        dim = MultiArrayDimension()
        dim.label = 'width'
        dim.size = 4
        dim.stride = 1
        msg.layout.dim = [dim]
        msg.layout.data_offset = 4
        msg.data = [v,v,0,s]
        self.motor_publisher.publish(msg)

    def sense_callback(self,pose_data,sensor_data):
        car_pos,omega = Environment.Point((pose_data.pose.position.x,pose_data.pose.position.y)),tf.transformations.euler_from_quaternion(pose_data.pose.orientation)[2]
        sensor_values = sensor_data.msg.data[:3]
        delta = Environment.Vector(car_pos,self.car.destination)
        s,c = self.env.encode_angle(delta.angle()-omega)
        dist = self.env.scale(delta.length(),0,self.env.arenas[self.car.connection]['max_delta'],0,1)
        self.car.phi = np.concatenate(([dist,s,c],self.env.scale(np.array(sensor_values),0,2,1,0)))

    def init_car(self):
        self.car.reset()
        self.env.compute_interaction(self.car)
        self.car.get_sensor_reading()
        self.env.randomize(self.rl.parameters['random_agent_position'],self.rl.parameters['random_destination_position'],self.car)
        self.env.set_max_steps(self.env_definition['max_steps'])
        self.gui.enable_trace(remove_traces=True)
        self.gui.set_run_select(self.gui.runs[0])
        self.gui.update_debug_info('[Training]\n')
        self.env.compute_interaction(self.car)
        self.rl.init_state_buffer(self.env,self.env_definition['dt'],self.car) # Necessary beacuse the simulator computes agent history, even when its disabled(when the history is set to 1)

    def simulate(self):
        self.gui.init_destination(True,self.car)
        terminals,terminal_states,physical_states,debug,log = self.rl.learn_step(self.env,self.env_definition['dt'],self.car)
        if debug is not None:
            self.gui.update_debug_info(debug)
            self.gui.update_graph(log['epoch'],log['avg_loss'],self.env_definition['graphs'][0])
            self.gui.update_graph(log['epoch'],log['total_reward'],self.env_definition['graphs'][1])
            self.gui.update_graph(log['epoch'],log['running_reward'],self.env_definition['graphs'][2])
        for i,term in enumerate(terminals):
            self.gui.update(term,terminal_states[i],draw_car=False,force_end_line=True)
        show_car = (self.car.epoch%100==0)
        self.gui.update(0,self.car.get_state(),draw_car=show_car)
        if show_car==True or len(terminals)>0:
            self.gui.refresh()

    def littlebot(self):
        state_buffer = np.copy(self.car.phi)
        action = self.rl.find_best_action(state_buffer,epsilon_override=0.0)
        v,s = self.rl.parameters['actions'][action]
        self.send_motor_command(v,s)
        self.gui.update(0,self.car.get_state(),draw_car=True)

    def control(self):
        selection = self.gui.get_run_select()
        self.gui.refresh()
        if not selection==self.run:
            print selection
            self.run = selection
        if self.run == self.gui.runs[0]:
            self.simulate()
        elif self.run == self.gui.runs[1]:
            self.littlebot()

if __name__=='__main__':
    rlcar = RLCAR()
    while(not rospy.is_shutdown()):
        rlcar.control()
