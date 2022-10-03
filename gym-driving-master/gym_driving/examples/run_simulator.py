# from curses import termname
from gym_driving.assets.car import *
from gym_driving.assets.dynamic_car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *
from gym_driving.controllers.controller import *

import time
import pygame, sys
from pygame.locals import *
import random
import cProfile
import IPython
import argparse

TIMESTEPS = 100000
SLEEP_DELAY = .0001
ACC_ACTION = 5.0
STEER_ACTION = 15.0
FPS = 30
SCREEN_SIZE = (500, 500)
SCREEN_COORD = (0, 0)

CAR_X = 0
CAR_Y = 0
CAR_ANGLE = 0
VEHICLES_X = [0, 0]
VEHICLES_Y = [-100, 100]
VEHICLES_ANGLE = [0, 0]
TERRAINS = []

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# RENDER_MODE = True
CONTROLLER_MODE = 'keyboard'
RENDER_MODE = True
# CONTROLLER_MODE = 'agent'
SCREENSHOT_DIR = None
# SCREENSHOT_DIR = 'screenshots'
# os.environ["SDL_VIDEODRIVER"] = "dummy"
"""
Controller Mode:
keyboard: Up/Down to accelerate, Left/Right to steer
xbox: Left stick up/down to accelerate, right stick left/right to steer
"""

def draw_box_coords(rectangle, screen, SCREEN_COORD):
    """
    Draws corners of input rectangle on screen,
    used for debugging.

    Args:
        rectangle: rectangle object
        screen: screen object
        SCREEN_COORD: 1x2 array, coordinates of center of screen
    """
    corners = rectangle.get_corners()
    for c in corners:
        pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
        pygame.draw.circle(screen, 0, pos, 5, 0)
    c = rectangle.get_pos()
    pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
    pygame.draw.circle(screen, 0, pos, 5, 0)

def simulate_driving_agent(search_horizon=3):
    """
    Simulates one trajectory controlled by the driving search agent.

    Args:
        search_horizon: int, number of timesteps in search horizon.

    Returns:
        counter: int, number of timesteps survived in trajectory. 
    """
    param_dict = {'num_cpu_cars': 5, 'main_car_starting_angles': np.linspace(-30, 30, 5), 'cpu_cars_bounding_box': [[-100.0, 1000.0], [-90.0, 90.0]]}
    pygame.init()
    fpsClock = pygame.time.Clock()
    if RENDER_MODE:
        screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Driving Simulator')
    else:
        screen = None
    simulator = DrivingEnv(render_mode=RENDER_MODE, screenshot_dir=SCREENSHOT_DIR, param_dict=param_dict)
    param_dict = {'search_horizon': search_horizon, 'driving_env': simulator}
    controller = Controller(mode='agent', param_dict=param_dict)
    
    done = False
    counter = 0
    simulator._reset()
    while counter < 100 and not done:
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = controller.process_input(simulator)
        # Steering only
        action = action[0]

        state, reward, done, info_dict = simulator._step(action)
        counter += 1
    return counter

    fpsClock.tick(FPS)

def run_driving_agent_experiment(num_experiments=50):
    """
    Simulates multiple trajectories controlled by the driving search agent.

    Args:
        num_experiments: Number of trajectories to run.
    """
    # search_horizons = [3, 5, 7]
    search_horizons = [5]
    result_dict = {}
    for search_horizon in search_horizons:
        print("Running Search Horizon: {}".format(search_horizon))
        scores, times = [], []
        param_dict = {'search_horizon': search_horizon}
        for _ in range(num_experiments):
            start = time.time()
            scores.append(simulate_driving_agent(search_horizon))
            end = time.time()
            times.append(end - start)
        result_dict[search_horizon] = {'mean_score': np.mean(np.array(scores)), \
            'mean_time': np.mean(np.array(times))}
        print("Results for search horizon = {}: ".format(search_horizon))
        print("Scores: ", scores)
        print("Times: ", times)
        print(result_dict[search_horizon])

    for search_horizon in search_horizons:
        print("Results for search horizon = {}: ".format(search_horizon))
        print(result_dict[search_horizon])


def simulate_manual_control(config_filepath=None):
    """
    Manually control the main car in the driving environment.
    
    Args:
        config_filepath: str, path to configuration file.
    """

    # print('config_filepath', config_filepath)
    # PyGame initializations
    pygame.init()
    fpsClock = pygame.time.Clock()
    if RENDER_MODE:
        screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Driving Simulator')
    else:
        screen = None

    if config_filepath is None:
        config_filepath = '../configs/config.json'
    controller = Controller(CONTROLLER_MODE)
    simulator = DrivingEnv(render_mode=RENDER_MODE, config_filepath=config_filepath)
    states, actions, rewards = [], [], []

    time.sleep(3)

    for t in range(TIMESTEPS):
        
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = controller.process_input(simulator)
        # Steering only
        # action = action[0]

        state, reward, terminate, info_dict = simulator._step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        fpsClock.tick(FPS)

        if t == TIMESTEPS - 1:
            states.append(state)

        # time.sleep(SLEEP_DELAY)

        if terminate:
            break


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        
        self.epsilon = 0.02
        self.learning_rate = 0.2
        self.x_divs = 20
        self.y_divs = 20
        self.theta_divs = 20
        self.num_actions = 4
        self.weights = np.zeros(((self.theta_divs*self.x_divs*self.y_divs + self.theta_divs*self.x_divs + self.theta_divs)*self.num_actions))
        self.discount = 1.0
        self.train_num_episodes = 5000
        self.test_num_episodes = 100
        self.upper_bounds = [390, 390, 360]
        self.lower_bounds = [-390, -390, 0]

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):

        x_step = (self.upper_bounds[0] - self.lower_bounds[0])/self.x_divs
        y_step = (self.upper_bounds[1] - self.lower_bounds[1])/self.y_divs
        theta_step = (self.upper_bounds[2] - self.lower_bounds[2])/self.theta_divs

        x_val = obs[0]
        y_val = obs[1]
        theta_val = obs[2]

        x_bin = (x_val - self.lower_bounds[0])//x_step
        y_bin = (y_val - self.lower_bounds[1])//y_step
        theta_bin = (theta_val - self.lower_bounds[2])//theta_step

        x_disc = math.floor(x_bin)
        y_disc = math.floor(y_bin)
        theta_disc = math.floor(theta_bin)

        # Thomas plis
        state_val = self.theta_divs*self.x_divs*y_disc + self.theta_divs*x_disc + theta_disc   
        feature_mat = np.zeros((self.theta_divs*self.x_divs*self.y_divs + self.theta_divs*self.x_divs + self.theta_divs, self.num_actions))
        
        feature_0 = np.copy(feature_mat)
        feature_0[state_val][0] = 1
        feature_0 = np.transpose(feature_0.flatten())

        feature_1 = np.copy(feature_mat)
        feature_1[state_val][1] = 1
        feature_1 = np.transpose(feature_1.flatten())

        feature_2 = np.copy(feature_mat)
        feature_2[state_val][2] = 1
        feature_2 = np.transpose(feature_2.flatten())

        feature_3 = np.copy(feature_mat)
        feature_3[state_val][3] = 1
        feature_3 = np.transpose(feature_3.flatten())

        return np.array([feature_0, feature_1, feature_2, feature_3])

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):

        s_0 = state[0]
        q_0 = np.dot(weights, s_0)

        s_1 = state[1]
        q_1 = np.dot(weights, s_1)

        s_2 = state[2]
        q_2 = np.dot(weights, s_2)

        s_3 = state[3]
        q_3 = np.dot(weights, s_3)

        q_list = [q_0, q_1, q_2, q_3]

        exploit_options = [0, 1]
        exploit_flag = np.random.choice(exploit_options, p = [epsilon, 1-epsilon])

        if(exploit_flag == 1):
            idx = np.argmax(q_list)      
        else:
            idx = np.random.choice(len(q_list))

        if idx == 0:
            action = np.array([0, 0])
        elif idx == 1:
            action = np.array([1, 0])
        elif idx == 2:
            action = np.array([0, 1])
        elif idx == 3:
            action = np.array([1, 1])

        return action    

    def get_idx_from_action(self, action):

        a_1 = action[0]
        a_2 = action[1]

        if a_1 == 0 and a_2 == 0:
            return 0

        elif a_1 == 1 and a_2 == 0:
            return 1

        elif a_1 == 0 and a_2 == 1:
            return 2

        elif a_1 == 1 and a_2 == 1:
            return 3

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):

        Q_cur = np.dot(weights, state[action])
        Q_new = np.dot(weights, new_state[new_action])
        const = learning_rate*(reward + Q_new - Q_cur)
        self.weights = weights + const*state[action]
        return self.weights

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, config_filepath):

        pygame.init()
        fpsClock = pygame.time.Clock()
        
        if RENDER_MODE:
            screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption('Driving Simulator')
        else:
            screen = None

        if config_filepath is None:
            config_filepath = '../configs/config.json'
        
        get_features = self.get_table_features
        weights = self.weights
        epsilon = self.epsilon
        learning_rate = self.learning_rate

        simulator = DrivingEnv(render_mode=RENDER_MODE, config_filepath=config_filepath)
        
        reward_list = []
        plt.clf()
        plt.cla()

        for e in range(self.train_num_episodes):

            state = simulator._reset()
            current_state = get_features(state)
            terminate = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            new_action_idx = self.get_idx_from_action(new_action)

            for f in range(1000):

                action = new_action
                action_idx = new_action_idx
                state, reward, terminate, info_dict = simulator._step(action)
                new_state = get_features(state)
                new_action = self.choose_action(new_state, weights, epsilon)
                new_action_idx = self.get_idx_from_action(new_action)
                weights = self.sarsa_update(current_state, action_idx, reward, new_state, new_action_idx, learning_rate,
                                            weights)
                current_state = new_state
                
                if terminate:
                    reward_list.append(-t)
                    break
                
                t += 1

            time.sleep(0.1)

        # self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig('train.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self):
        get_features = self.get_table_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    
    args = parser.parse_args()
    train=int(args.train)
    config_filepath = args.config

    agent = sarsaAgent()
    # np.random.seed(0)

    if(train):
        agent.train(config_filepath)
    else:
        print(agent.test())
    
    # simulate_manual_control(config_filepath)

