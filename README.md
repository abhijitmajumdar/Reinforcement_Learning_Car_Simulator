# Reinforcement Learning Car Simulator
A multi agent multi arena car simulator oriented towards Reinforcement Learning. Provides functionality of defining multiples similar or different cars(agents) in single or multiple arenas(environment), with each car having sensing capability with the use of obstacle detector similar to a laser range-finder. Simulator is capable of running multiple instances simultaneously while logging all data separately. Example with the use of Deep Q Learning (DQN) is provided for single agent in different arenas.

Example of learning an arena using DQN:
![Learning](/Images/rlcarsim_demo3.gif?raw=true "Learning")
<br/><br/><br/><br/>
Examples of learned policy evaluation:<br/>
<img src="/Images/rlcarsim_demo5.gif" width="35%"> <img src="/Images/rlcarsim_demo4.gif" width="35%">
<img src="/Images/rlcarsim_demo1.gif" width="35%"> <img src="/Images/rlcarsim_demo2.gif" width="35%">
<br/>Other examples of arenas:<br/>
<img src="/Images/rlcarsim_user_mama_boxh.png" width="55%"> <img src="/Images/rlcarsim_user_multiagent_room.png" width="28%"><br/>

## Dependencies
- Python built-in modules:
    - os
    - math
    - random
    - time
    - Tkinter (since the code is written in Python 2)
    - argparse
    - shutil
- External Dependencies (Can be install on most systems using `pip install <package>`):
    - numpy
    - keras (I use Tensorflow backend)
    - tensorflow
    - configobj

## How to run
Clone the repository, move into the directory, and run the code:
```sh
$ git clone https://github.com/abhijitmajumdar/Reinforcement_Learning_Car_Simulator.git
$ cd Reinforcement_Learning_Car_Simulator
$ python rlcarsim.py --control dqn
```

#### Other examples:
Almost every setting can be configured using a configuration file. Sample configuration files are provided in the `Configurations` folder representing different settings the simulator can be used for.

Run with predefined velocities and steering, with user controlling all cars(w/a/s/d), with single or multiple cars:
```sh
$ python rlcarsim.py --control user --config Configurations/config_singleagent.ini --arena BOX
$ python rlcarsim.py --control user --config Configurations/config_multiagent.ini --arena ROOM
$ python rlcarsim.py --control user --config Configurations/config_singleagent.ini --arena SHAPE
$ python rlcarsim.py --control user --config Configurations/config_multiagent.ini --arena H
$ python rlcarsim.py --control user --config Configurations/config_multiagent.ini --arena CIRCLE
```

Run with user control, on multiple cars and multiple arenas:
```sh
$ python rlcarsim.py --control user --config Configurations/config_multiagent_multiarena.ini --arena BOX,H
$ python rlcarsim.py --control user --config Configurations/config_multiagent_multiarena.ini --arena BOX,H,CIRCLE
$ python rlcarsim.py --control user --config Configurations/config_multiagent_multiarena.ini --arena BOX,CIRCLE
```

Train a model using DQN in the BOX environment with an obstacle (The training and validation can be switched in the GUI as well, as described later):
```sh
$ python rlcarsim.py --control dqn --config Configurations/config_singleagent.ini --arena BOX
```
Test the model in a different(BIGBOX) environment with an obstacle by loading weights *rlcar_epoch_00600* in run 5(folder inside weights directory):
```sh
$ python rlcarsim.py --control dqn --config Configurations/config_singleagent.ini --arena BIGBOX --run_only --load_weights weights/5/rlcar_epoch_00600
```

Train a model using DQN in the TRACK(which uses path creator to convert path points into polygon points, enabled in the config file) environment without obstacles:
```sh
$ python rlcarsim.py --control dqn --config Configurations/config_track.ini --arena TRACK
```


## Working
The goal was to create a customized environment, where a car like object can be trained to maneuver and reach its destination while avoiding obstacles with the help of sensors on the car. A simulator is built with a Graphical User Interface (GUI) which shows the track and the car like objects along with obstacles. The project contains four modules in the Simulator folder `Environment.py`, `RL.py`, `GUI.py` and `Utils.py` which define different classes and methods used by the main file `rlcarsim.py`. This file gives example usage of each module and can be customized as per the application.

Almost everything used in the simulator is configurable using a configuration file. Sample configuration files for different use of the simulator are provided in the `Configurations` folder, which can be referred to construct a different setup if needed. More details about the configuration files later. Using the argument `--control user` or `--control multi` with the configuration files gives the user the ability to test the arenas, obstacles and the interactions of the cars with them form the terminal. When starting such a user based control, the cars may not be visible initially; in such cases press `r` in the terminal to reset the environment.

The simulator is oriented towards *Reinforcement Learning (RL)*, and an example use of RL with the simulator can be run using the argument `--control dqn` to run the `rlcarsim.py` file which uses a Deep Q Network (DQN) to learn for a car to reach its goal using the default configuration file. The implementation of the DQN algorithm is made in the `RL.py` file. The learned weights can be loaded using the command line argument `--load_weights <path_to_weights>`, which can be used to resume training or load a checkpoint to observe evaluation using additional argument `--run_only`.

One of the key advantages of the simulator is being lightweight, multiple instances can be run at the same time. One need not make copy of the source files, since the simulator logs each run separately, and hence has no issue of overwriting. As a result all instances can be run from the same folder. This is useful in cases where one needs to test different configurations to find the best one. An example of 4 instances (2 same and 2 different) of the simulator on an *i5-6600 + GTX1080* is shown below.

![Simultaneous instance runs](/Images/rlcarsim_mi.png?raw=true "Simultaneous instance runs")

## Modules
##### Environment
The environment and the car objects are defined in this file and computes interactions between all physical objects in the environment using vector geometry. The car dynamics are based on the Bicycle model of a car. The car objects save properties related to the car, like the state, the sensor values and other scoring and configuration settings defined in the configuration file. The destination is set individually for each car, so that in a multi agent setup, different destinations for each agent can be assigned. The environment updates the interactions and modifies the state of the car and sensors accordingly. The sensors on the car are shown with a red line, indicating the range of sensing for each sensor. They act as distance sensors, providing the distance to the nearest obstruction to the *sensing ray*. Practically these sensors can be thought of as either sonar sensors, infrared sensors or even laser range finders.

The module uses the user argument to figure out which environment to interact with, which may be a multi arena setup. In case of a multi-arena setup, each agent is assigned a *connection* value which specifies the arena they belong to. There are two methods in which this module interprets the arena, one in which the points defining a polygon for an arena may be defined, and the other where the points represent a path along which a track of *track_width* width is computed. In case of a track like arena, the user needs to define only path along which the track is needed, and set *path_creator* to *True*, which creates a track. This is configurable and can be used to form different types of arenas. **When using track like arena, make sure random_agent_position and random_agent_position are set to False in the config file to ensure proper learning.**

The configurable parameters to define if the agent(s) and destination(s) are reset to random positions are available and make sure the reset positions are valid.

##### GUI
The GUI is constructed using *Tkinter* library. It defines a window in which the cars and arenas are simulated from a top view perspective on the left side of the window, while the right side is organized for debugging, options and graph plotting. The resolution of the GUI is configurable and the module fits the arenas to best fit everything proportionally. The right side contains buttons to select between the process of learning verses run only(testing), along with debugging information like epoch, scores, loss, weight files etc. There is a graph representation, the contents of which can be switched using the provided buttons, labels of which are defined in the configuration file. The *dispaly_dk* parameter defines how slow the display is updated to view a conceptual result. During evaluation, mouse click on the GUI changes the destination to the closest valid location.

##### RL
This module defines classes as examples to be used for RL applications. *ReplayMemory* and *QLearning_NN* are base classes, defined to allow other algorithms to build on top of. A sample *DQN* class if defined which performs training and testing on a car agent.

The action-value neural network approximator can be defined by specifying the number of neurons in each hidden layer and the activation function in the configuration file under the tag *[Network]*. Similarly, the RL parameters used like *epsilon* used in the *epsilon-gredy* policy, *gamma* as the discount factor, *alpha* as the learning rate and *C* as the *target network update frequency*, might all be defined in the configuration file as well under the tag *[Reinforcement Parameters]*. The permissible actions are also defined here under the tag *[Actions]*. A *log directory* is used to specify the directory under which all log files are stored, including training checkpoints(weights), configuration file used for the run and a log file containing details of all statistics for the run. A separate directory under the specified log directory is created for each run, to ensure no overwriting is performed, and all runs are logged, which can be used to compare later.

The algorithms, methods and parameters used are largely influenced by the following:
- Human-level control through deep reinforcement learning. https://www.nature.com/articles/nature14236 (With the difference in interpreting epsilon value\*)
- Reinforcement Learning course by David Silver. http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
- https://github.com/harvitronix/reinforcement-learning-car

\*~~I think epsilon=0.7 in the epsilon-greedy policy means, be greedy to follow the learned policy(determined by the action-value function) 70%(epsilon) of the time. However the epsilon used in the paper is in a reverse manner, and hence propagates from 1.0 to 0.1, while the epsilon in this project propagates from 0.0 to 0.9.~~ After inspecting the paper, I realized that the epsilon-greedy policy is with respect to exploration, and hence it makes sense to use epsilon as the probability to use random actions. ~~However, I keep the use of epsilon inverted in this project.~~ Epsilon now varies as convention

##### Utils
Consists of methods to parse the command line arguments and the configuration file. Also performs checking and creating new directory for current run (indexed by run iteration) inside the log directory.

## Configuration
A `*.ini` file may be defined to configure the setup of the simulator and other parameters used. Sample configuration files are provided in the `Configurations` folder for reference. The default configuration path points to `Configurations/config.ini` file but might be specified explicitly as a command line argument as `--config <path_to_config_file>`. The file expects `name=value` format configurations categorized under sections and subsections. For a quick review of how to configure a configuration file, use the provided examples or have a look at these links:
- https://en.wikipedia.org/wiki/INI_file
- https://docs.python.org/3/library/configparser.html (Though I dont use the Python built-in parser, since I use subsections and Python data-types in the config file)

## Changes
- added MVEDQL class
- obstacles are now defined as sub parts of an arena
- configuration file can specify one or many arenas to be used, however it can be overridden by a command line argument `--arena`
- changed epsilon greedy implementation to follow convention (high to low) insted of (low to high), though the meaning remains same.
- A check agent connection method is introduced to raise error if mis-configured
- NN architechture configurable in config file
- Each agent now has its own destination and connection to arena configured into its object definition
- Environment class can now configure multiple arenas defined by env_select
- Environment now has a randomize method with selectable agent position and/or destination to valid points.
- GUI can also initialize multiple arenas configured through env_select
- GUI is scaled based on the areas defined automatically to fit all into space available
- GUI resolution is now configurable
- Currently mouse click changes destination of all agents to that point
- Program argument `--env` can take in multiple arena names as comma seperated values to define a multi arena environment
- Added a configuration `.ini` file to configure `everything`
- a new folder is created inside the log directory specified in the config file each run, to avoid overwriting the previous saved weights and have a configuration log
- Added basic Q learning class and DQN implementation class making it modular
- Colorized cars, goals and obstacles
- Corrected destination as terminal state
- Added a parameter replay_start_at which decides when the learning starts even before completely filling up replay, hence allowing larger replay
- Realized epsilon-greedy was with respect to exploration, however its inverted here.
- Found the cars to run much better with the following parameters: epsilon from 0.0 to 0.9 with steps of 0.0015, discount factor gamma:0.99, reward system of terminals as -1(collision,timeup), +1(destination) and non-terminals as -1(car score reduces), +1(car score increases). While learning the NN diverges as new values are propagated and then converges.
- Increased replay memory, so that all history can be used
- Added a feature(parameters->random_car_position) to randomize car position and orientation in the start region while training.
- Removed 'v' and 'gamma' from car definitions, they made no sense and not used anywhere
- Added a function make_parameter_changes and arguments to the parser, to modify car sensors with arguments --sensor_length and --add_more_sensors along with the random start argument --random_car_position.
- Added track_width and start_angle to environment variables to be used for random start of cars.
