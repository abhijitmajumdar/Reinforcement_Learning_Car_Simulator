# Reinforcement Learning Car Simulator
A project to show how RL (specifically Q-learning) can be applied to make a car learn to navigate through an arbitrary track without colliding, given a couple of distance sensors.

![Learning demo](/Images/rlcarsim_demo.gif?raw=true "Demo")
Learning

![Run only demo](/Images/rlcarsim_learned_demo_long.gif?raw=true "Demo")
![Run only demo](/Images/rlcarsim_learned_demo.gif?raw=true "Demo")
Learned

## Dependencies
- Simulation of dynamics and RL:
    - Numpy
    - Math
    - Random
    - Keras (I use Tensorflow backend)
- GUI and plotting:
    - Tkinter
    - Time
    - Matplotlib (Only used when plotting external to the GUI)

## How to run
Clone the repository, move into the directory, and run the code:
```sh
$ git clone https://github.com/abhijitmajumdar/Reinforcement_Learning_Car_Simulator.git
$ cd Reinforcement_Learning_Car_Simulator
$ python rlcarsim.py --control nn
```

#### Argument examples:
Run with predefined velocities and steering, with user controlling all cars(w/a/s/d) or with RL NN:
```sh
$ python rlcarsim.py --control static
or
$ python rlcarsim.py --control user
or
$ python rlcarsim.py --control nn
```
Running a particular learned weight at epoch 675 or loading all saved weights iteratively:
```sh
$ python rlcarsim.py --control nn --run_only --load_weights ./weights/rlcar_epoch_00675
or
$ python rlcarsim.py --control nn --run_only --load_weights all
```
Selecting a track among the list of tracks defined in rlcarsim.py, adding a random seed value and randomly initializing car start position:
```sh
$ python rlcarsim.py --control nn --track SE --random_seed 33 --random_car_position
```
Setting predefined sensor configurations:
```sh
$ python rlcarsim.py --control nn --sensor_length short
or
$ python rlcarsim.py --control nn --sensor_length long
or
$ python rlcarsim.py --control nn --sensor_length long --add_more_sensors
```
Better visualization of the loss and scores (After/during training):
```sh
$ python plot_log.py
```
The training and run can be switched in the GUI as well, as described later.


## Working
The goal was to create a custom track, where a car like object can be trained to maneuver the track with the help of sensors on the car. A simulator is built with a Graphical User Interface (GUI) which shows the track and the car like objects. The project contains three files *Environment.py, RL.py and GUI.py* which define different classes used by the main file *rlcarsim.py*. The main file contains methods for different controls of the cars: static control, user control and RL control.

When the program is run with *--control nn* arguments, the RL neural network training begins. Though all cars are initialized in the GUI, only the first car object is simulated to gather data for training the network. At each iteration, the car is moved with action taken based on the epsilon-greedy policy, and corresponding data for old state, action taken, reward gained and new state are stored in the *replay memory*. The probability of choosing action by the learned action-value function approximator verses choosing random action, the epsilon value, is incremented after each epoch. An epoch/episode is defined as the a run of the agent(car) until it terminates, when a terminal state is reached, which in this case may be either a collision, a timeout or a destination reached case.

Initially the car accumulates replay data without learning, while filling up the replay bank. Once the memory is filled upto a size defined by parameter *replay_start_at*, every successive iteration also involves random selection of a *minibatch* of samples from this replay memory and training the neural network based action-value function approximator using back propagation. While training, the GUI is updated with a run of the car every 100th episode. This is to encourage faster processing in the background, since a visual update is the biggest factor for a bottleneck in processing (due to the delay required to visualize the frame before update). However, the paths taken by the agent during this background processing can be seen updated on the GUI, by gray lines. It is interesting to note how these lines progress as the agent starts to learn, and follow through the track.

For the batchsize set in the project, the agent usually takes approximately 700 epochs to fill up the replay memory and start training. For this period of time, the actions taken by the agent are random. Convergence to the destination in the *SS* track is usually achieved in 1400-1700 epochs. If the agent is not able to learn to reach the destination, it is advisable to restart the training from scratch or make changes to the parameters. The divergence or convergence nature can be observed from the *Average Loss* and *Cross Score* graphs plotted every 5 epochs in the GUI. The graph views can be switched using the buttons in the GUI (It takes 10 epochs to update though, will fix that later).

Example of converging network

<img src="/Images/rlcarsim_convergence.png" width="30%"> <img src="/Images/rlcarsim_convergence_plot.png" width="40%"> <img src="/Images/rlcarsim_convergence_trace.png" width="50%">

The convergence of the network might also be realized by observing the trace lines(gray lines indicating the paths taken by the agent during training) reaching the end of the track, the average score achieved by the car is approximately 41(for the SS track) and the run resulting in destination, all of which can be viewed in the GUI. At this point one might want to observe how the learned model performs if we followed the action-value function entirely (epsilon=100%). The GUI buttons *Learning* and *Run only* can be used to switch between training and testing the learned model. In the run-only method, epsilon is set to 1.0, meaning that the learned model is followed without any randomness, and all cars are simulated to complete the track. If however a particular learned weight needs to be observed, the simulation can be run by specifying the arguments to load the particular weight, an example of which is shown in the section *Argument examples* using the *--load_weights* and *--run_only* arguments.

When the learning starts, the agent tries to avoid colliding as the value of epsilon slowly increases. Initially the policy followed by the agent is that of a uniform normal distribution to follow any action, and hence the loss computed by the neural network training is low. As the agent learns and modifies its policy towards optimal behavior, the loss increases, since new values are to be back-propagated to for the neural network weights. However, this loss keeps increasing since, now the agent tries to follow this improved policy and hence accumulate more rewards, which propagates through the network and sets the value of weights different from what it was initially. As a result the loss keeps increasing initially while the agent learns and follows the improved policy. After some time this loss value starts reducing, which is when the agent has explored the maximum reward possible and now fits the NN weights. The cars seem to already learn to navigate the track, however if we wait for the network loss to minimize to ~0, the cars perform much better navigating the track.

It was also observed that with smaller sensor range, there are scenarios where the value of the sensors are not sufficient to describe the state of the system completely. For example with smaller sensor ranges, the car cannot tell if its at the center of the track, or which direction the tracks are running along. However even with the smaller sensor ranges, the cars recognize this incomplete state situation and learns to find the edges of the track to keep in contact until it reaches its destination. Cars with large sensor ranges can be observed to travel fast through the middle of the track when its straight and slower and closer to the curbs around a corner. This behavior is analogous to that in a motorcycle race, since the dynamic model used for the car is a bicycle model. This can be simulated using the argument *--sensor_length short* or *--sensor_length long* to the run.

Example of diverging(most often) network with bad parameters:

<img src="/Images/rlcarsim_divergence.gif" width="30%"> <img src="/Images/rlcarsim_divergence_trace.png" width="55%">

More tests can be simulated using the arguments *--add_more_sensors* and *--random_car_position* which alter the sensor configurations and randomly initialize the start state of the car respectively.

Note: the loss value can build up to large values(~1000000), but then reduce eventually. A random seed value of 427 produced good results.

##### Car definitions
The cars are defined in a list *cars* which contain the definition of a car with details like initial state, length, width, initial and maximum velocity and steering angles, sensors with maximum range and orientation of each with respect to the car. Orientation of the sensors are with respect to the cars' orientation, either of which is in radians.

##### Track definitions
The track is defined by assigning coordinate points *P(x,y)* in the *rlcarsim.py* file. Methods are later defined which add width and construct a track from these coordinates. Multiple tracks can be defined inside the dictionary *track*, from which a particular track may be selected at runtime using keyword arguments to the program.

##### Parameters
The parameters that govern the reinforcement algorithm are defined in a dictionary *rl_parameters* with the following:
- actions: List of actions possible for a car to take, in the form of tuples of velocity and steering angles
- epsilon: The initial probability which which an agent selects an action governed by the current policy over some random action. Application of epsilon-greedy action selection policy
- max_epsilon: The maximum value of epsilon which the algorithm increments after each epoch.
- epsilon_step: The step by which the algorithm increments the value of epsilon
- gamma: The discount factor. Lower values make the agent tend to achieve recent upcoming rewards, compared to a larger value of gamma which encourages the agent to be far sighted and work towards expected future rewards.
- lr_alpha: the learning rate for the neural network
- leak_alpha: The alpha value with which LeakyReLU *leaks*
- max_steps: The maximum number of steps allowed for a car to reach its goal before terminating it. Used to avoid car going round and round in circles
- collision_reward, timeup_reward, destination_reward: rewards for respective states
- buffer_length: The size of replay memory
- replay_start_at: The size of replay memory to fill up before training starts
- batchsize: Size of batch to process weight update, varied based on CPU/GPU resources available
- minibatchsize: Size of batch to update weight at each step. Large values learn more but are slower to process
- state_dimension: Size of inputs to the action-value function(NN) -> number of sensors on the 1st car
- random_car_position: Initialize the position and orientation of the car randomly while starting a new epoch during training to explore all possible start points.
Also
- dt: The interval after which the state of a car is updated/controlled

##### Environment.py file
The track environment and the car objects are defined in this file. They use Vector and Point classes to compute interactions between the car, the sensor and the track using vector geometry. The car dynamics are based on the Bicycle model of a car(not the Ackerman model). The car has update methods to update its state in the environment based on an action taken in one time step(dt). The environment updates the interactions and modifies the state of the car and sensors accordingly. The sensors on the car are shown with a red line, indicating the range of sensing of each sensor. They act as distance sensors, providing the distance to the nearest obstruction to the *sensing ray*. Practically these sensors can be thought of as either sonar sensors, infrared sensors or even laser range finders. Methods in this class have been influenced by:
- http://geomalgorithms.com/a05-_intersect-1.html#intersect2D_2Segments()
- https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
- Robotics, Vision and Control by Peter Croke - 4.2 Car-like Mobile Robots
- https://www.cs.cmu.edu/~quake/robust.html
- https://stackoverflow.com/questions/17581738/check-if-a-point-projected-on-a-line-segment-is-not-outside-it
- http://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
The car model provided is not very accurate, since drag effects are not considered, and update to velocities and steering angles are immediate. Though these might be easily fixed later, for now the RL algorithm is being tested on a simpler model first. Another fact to consider is that the sensors simulated detect obstruction to a ray, however, practical sensors usually have a cone like sensing area.

##### RL.py file
This contains the neural network based Q-learning algorithm class definition. The parameters previously defined are passed onto an object of this class, which defines methods to create a neural network, load weights, take epsilon-greedy action, determine rewards, train the neural network, and check for terminal states.

The action-value function used to determine the action with the best value(reward + future rewards), is what we want to learn, to follow the optimal path to the goal(in this case avoiding obstacles and reaching destination, in time). For a small finite set of states(input values to this action-value function), convergence is guaranteed. However in this example, the states are represented by the sensor values, which are considered continuous. Hence we use a neural network function approximator to predict the action-value for each state. As per my experiments, I found LeakyReLU to work best as the activation function for the neural network. As for the architecture of the neural network, trying out different network size and width can be experimented with. Also initial weights are not randomized, but set fixed to compare the performance of different algorithm parameters reliably.

Determining a reward function is a major factor for the algorithm to learn the desirable behavior. For example, initially the lack of a timeout reward(negative) caused the car to learn to run in circles for ever, since that way it avoided any collision and hence obtained higher value. ~~After trying different rewards for a normal running state of the car, I fixed on the car score, which is the distance traveled by the car along the track, encouraging the agent maximize the reward by heading towards the goal. However, using this method meant that the discount factor *gamma* needed to be small to indicate the agent to strive for short sighted maximum rewards. Iterations with a destination reward(positive) did not produce good results, and hence is ignored in the current algorithm, nevertheless can be used if required with minor changes.~~ I tried out different reward functions evaluating the performance, and fixed the one where reward for collision and timeout were -1, for destination was +1, all of which were terminal states. Also to encourage the car to move forward, a discounted reward of +1 was given if moved forward and -1 otherwise. This simplified the model to try to accumulate more rewards over the long run since we set the value of discount factor *gamma* as 0.99. These configurations were inspired from the paper referred below and the results with these configurations were much consistent than before.

The algorithms, methods and parameters used are largely influenced by the following:
- Human-level control through deep reinforcement learning. https://www.nature.com/articles/nature14236 (With the difference in interpreting epsilon value\*)
- Reinforcement Learning course by David Silver. http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

\*~~I think epsilon=0.7 in the epsilon-greedy policy means be greedy to follow the learned policy(determined by the action-value function) 70%(epsilon) of the time. However the epsilon used in the paper is in a reverse manner, and hence propagates from 1.0 to 0.1, while this epsilon in this project propagates from 0.76 to 0.98.~~ After inspecting the paper, I realized that the epsilon-greedy policy is with respect to exploration, and hence it makes sense to use epsilon as the probability to use random action. However, I keep the use of epsilon inverted in this project.

##### GUI.py file
The GUI is constructed using *Tkinter* library. It defines a window in which the cars and track is simulated in a top view fashion on the left side of the window, while the right side is organized for debugging and options. The track defined in the main file is automatically fitted into the space alloted to the left side. This means that if a much larger track is defined, the visual representation will be smaller. The right side contains buttons to select between the process of learning verses run only, along with debugging information like epoch, scores, loss, weight files etc. There is a graph representation, the contents of which can be switched using the provided buttons for *Average loss, Final score and Cross scores* respectively. The graphs can be better visualized using the provided *plot_log.py* file, an example to use this can be found in the section *Argument examples* above.

## Changes
- Added a parameter replay_start_at which decides when the learning starts even before completely filling up replay, hence allowing larger replay
- Realized epsilon-greedy was with respect to exploration, however its inverted here.
- Found the cars to run much better with the following parameters: epsilon from 0.0 to 0.9 with steps of 0.0015, discount factor gamma:0.99, reward system of terminals as -1(collision,timeup), +1(destination) and non-terminals as -1(car score reduces), +1(car score increases). While learning the NN diverges as new values are propagated and then converges.
- Increased replay memory, so that all history can be used
- Added a feature(parameters->random_car_position) to randomize car position and orientation in the start region while training.
- Removed 'v' and 'gamma' from car definitions, they made no sense and not used anywhere
- Added a function make_parameter_changes and arguments to the parser, to modify car sensors with arguments --sensor_length and --add_more_sensors along with the random start argument --random_car_position.
- Added track_width and start_angle to environment variables to be used for random start of cars.

## To-do
- Make the change in graph to button click immediate
- Add other RL methods
- Change the dynamics to be more realistic (acceleration and delta-steering model)
- Implement an actual Ackerman model to simulate more realistic car drive, compare to the close-to-curb-around-corners behavior that it has currently.
