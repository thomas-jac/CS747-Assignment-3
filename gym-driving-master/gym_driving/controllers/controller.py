from gym_driving.controllers.xboxController import *
# from gym_driving.agents.driving_agent import *

import numpy as np
import math
import matplotlib.pyplot as plt

class Controller:
    def __init__(self, mode='keyboard', param_dict=None):
        """
        Initializes controller object to unify input interface.

        Args:
            mode: str, determines mode of input control.
                Must be in ['keyboard', 'xbox'].
            param_dict: dict, parameters to pass into controller.
        """
        self.mode = mode
        if mode == 'keyboard':
            pass
        elif mode == 'xbox':
            self.xbox_controller = XboxController()
        else:
            raise NotImplementedError


        self.epsilon = 0.02
        self.learning_rate = 0.2
        self.x_divs = 20
        self.y_divs = 20
        self.theta_divs = 20
        self.num_actions = 4
        self.weights = np.zeros((self.x_divs*self.y_divs*self.theta_divs*self.num_actions))
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [378, 378, 180]
        self.lower_bounds = [-378, -378, -180]

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
        feature_mat = np.zeros((self.theta_divs*self.x_divs*self.y_divs, self.num_actions))
        
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
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''


    def process_RL(self, state, weights, epsilon):

        """
        Process an input from the keyboard.

        Returns:
            action: 1x2 array, steer / acceleration action.
        """

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

    def train(self, simulator):
        
        get_features = self.get_table_features
        weights = self.weights
        epsilon = self.epsilon
        learning_rate = self.learning_rate
        

        reward_list = []
        plt.clf()
        plt.cla()

        for e in range(self.train_num_episodes):

            simulator._reset()
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)

            while not done:

                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1

        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    def process_input(self, env):
        """
        Process an input.

        Args:
            env: environment object, used for agent.

        Returns:
            action: 1x2 array, steer / acceleration action.
        """

        if self.mode == 'manual':
            action = self.process_RL()
        elif self.mode == 'keyboard':
            action = self.process_keys()
        elif self.mode == 'xbox':
            action = self.process_xbox_controller()
        return action

    def process_keys(self):
        """
        Process an input from the keyboard.

        Returns:
            action: 1x2 array, steer / acceleration action.
        """
        action_dict = {'steer': 0.0, 'acc': 0.0}
        steer, acc = 1, 1
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            acc = 2
        elif keys[pygame.K_DOWN]:
            acc = 0
        if keys[pygame.K_LEFT]:
            steer = 0
        elif keys[pygame.K_RIGHT]:
            steer = 2
        action = np.array([steer, acc])
        return action

    def process_xbox_controller(self):
        """
        Process an input from the Xbox controller.

        Returns:
            action: 1x2 array, steer / acceleration action.
        """
        action_dict = {'steer': 0.0, 'acc': 0.0}
        left_stick_horizontal, left_stick_vertical, \
        right_stick_horizontal, right_stick_vertical = \
                        self.xbox_controller.getUpdates()
        steer = np.rint(right_stick_horizontal) + 1
        acc = -np.rint(left_stick_vertical) + 1
        action = np.array([steer, acc])
        return action