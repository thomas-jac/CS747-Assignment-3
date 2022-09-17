# Driving Simulator

Most of the source code for the driving environment has been taken from https://github.com/WesleyHsieh/gym-driving/tree/master/gym_driving.
To use this driving simulator, you will need to install PyGame and OpenAI Gym.

### Steps to install and run driving simulator (manually)

* Create a new virtual environment (this can be useful, because there are occasionally errors in installing some packages when there is a version of numpy already installed).
> git clone https://github.com/WesleyHsieh/gym-driving.git

> cd gym-driving

> pip install -e . # This step installs all packages needed, except gym

> cd ..

> git clone https://github.com/openai/gym.git

> cd gym

> pip install -e .

> cd ../gym-driving/gym_driving/examples

> python run_simulator.py

This should display a pygame window and you can use your keyboard to move the car. By default, it uses the config "config.json" in gym-driving/gym_driving/configs.

## Configuration

generate_config.py: Generates configuration file that the environment reads from. Allows configuration of most features in the simulator, including state space, action space, number of and positions of CPU cars, track placement, etc. 

configs/: Folder for storing configuration files.

The default demo script (run_simulator.py) reads from the configuration script in configs/config.json. To change the config it runs from, use the command 

> python run_simulator.py --config new_config.json



