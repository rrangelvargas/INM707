import pygame
from enum import Enum
import sys
import os
import random
import numpy as np
import time
class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    COLLECT = 4
    RECHARGE = 5
    TRANSMIT = 6

class Robot():
    def __init__(self):
        self.position = (0, 0)
        self.battery = 100
        self.rock_count = 0
        self.holding_rock = False
        self.action = Actions.RIGHT

class Mars_Environment():
    def __init__(
        self, size,
        max_epsilon = 1,
        min_epsilon = 0.05,
        decay_rate = 0.0001,
        alpha = 0.1,
        gamma = 0.9,
        no_episodes = 1000,
        max_steps = 100
    ):
        self.size = size
        self.window = 800
        self.display = None
        self.clock = None
        
        self.epsilon = max_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.no_episodes = no_episodes
        self.max_steps = max_steps
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

        self.q_table = []
        self.rewards = []
        self.robot = Robot()
        self.rocks = [(1,2), (3,3), (2,4)]
        self.transmiter_stations = [(4,4)]
        self.cliffs = [(2,3), (1,1)]
        self.uphills = [(0,4), (2,0)]
        self.downhills = [(3,0), (0,2)]
        self.batery_stations = [(4,3)]
        self.initialize_q_table()
        self.fill_rewards()
        self.render()

    def initialize_q_table(self):
        self.q_table = [[0 for _ in range(len(Actions))] for _ in range(self.size * self.size)]

    def reset(self):
        self.robot.position = (0, 0)
        self.robot.action = Actions.RIGHT
        self.robot.battery = 100
        self.robot.holding_rock = False
        self.initialize_q_table()
        self.fill_rewards()

    def fill_rewards(self):
        self.rewards = [[0 for _ in range(self.size * self.size)] for _ in range(self.size * self.size)]

        for rock in self.rocks:
            self.rewards[rock[0] + self.size*rock[1]][rock[0] + self.size*rock[1]] = 20
        
        for transmiter in self.transmiter_stations:
            self.rewards[transmiter[0] + self.size*transmiter[1]][transmiter[0] + self.size*transmiter[1]] = 50

        for battery in self.batery_stations:
            self.rewards[battery[0] + self.size*battery[1]][battery[0] + self.size*battery[1]] = 20

        for cliff in self.cliffs:
            if cliff[0] > 0:
                self.rewards[cliff[0]-1 + self.size*cliff[1]][cliff[0] + self.size*cliff[1]] = -50
            if cliff[0] < self.size-1:
                self.rewards[cliff[0]+1 + self.size*cliff[1]][cliff[0] + self.size*cliff[1]] = -50
            if cliff[1] > 0:
                self.rewards[cliff[0] + self.size*(cliff[1]-1)][cliff[0] + self.size*cliff[1]] = -50
            if cliff[1] < self.size-1:
                self.rewards[cliff[0] + self.size*(cliff[1]+1)][cliff[0] + self.size*cliff[1]] = -50
        
        for uphill in self.uphills:
            if uphill[0] > 0:
                self.rewards[uphill[0]-1 + self.size*uphill[1]][uphill[0] + self.size*uphill[1]] = -5
            if uphill[0] < self.size-1:
                self.rewards[uphill[0]+1 + self.size*uphill[1]][uphill[0] + self.size*uphill[1]] = -5
            if uphill[1] > 0:
                self.rewards[uphill[0] + self.size*(uphill[1]-1)][uphill[0] + self.size*uphill[1]] = -5                
            if uphill[1] < self.size-1:
                self.rewards[uphill[0] + self.size*(uphill[1]+1)][uphill[0] + self.size*uphill[1]] = -5

        for downhill in self.downhills:
            if downhill[0] > 0:
                self.rewards[downhill[0]-1 + self.size*downhill[1]][downhill[0] + self.size*downhill[1]] = 5
            if downhill[0] < self.size-1:
                self.rewards[downhill[0]+1 + self.size*downhill[1]][downhill[0] + self.size*downhill[1]] = 5
            if downhill[1] > 0:
                self.rewards[downhill[0] + self.size*(downhill[1]-1)][downhill[0] + self.size*downhill[1]] = 5
            if downhill[1] < self.size-1:
                self.rewards[downhill[0] + self.size*(downhill[1]+1)][downhill[0] + self.size*downhill[1]] = 5

    def process_image(self, filename, scale_factor):
        path = os.path.join(os.getcwd(), "images", filename)
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (int(self.grid_size * scale_factor), int(self.grid_size * scale_factor)))
        image = image.convert_alpha()
        return image

    def load_images(self):
        _robot = self.process_image("robot.png", 0.6)
        _rock = self.process_image("rock.png", 0.6)
        _transmiter = self.process_image("antenna.png", 0.6)
        _cliff = self.process_image("cliff.png", 0.6)
        _uphill = self.process_image("uphill.png", 0.6)
        _downhill = self.process_image("downhill.png", 0.6)
        _battery = self.process_image("battery.png", 0.6)
        return _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery

    def blit_objects(self, image, objects, offset=0.2):
        for obj in objects:
            pos = (obj[0] * self.grid_size + self.grid_size * offset, obj[1] * self.grid_size + self.grid_size * offset)
            self.display.blit(image, pos)

    def render_images(self, _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery):
        self.blit_objects(_robot, [self.robot.position])
        self.blit_objects(_rock, self.rocks)
        self.blit_objects(_transmiter, self.transmiter_stations)
        self.blit_objects(_cliff, self.cliffs)
        self.blit_objects(_uphill, self.uphills)
        self.blit_objects(_downhill, self.downhills)
        self.blit_objects(_battery, self.batery_stations)

    def render(self):
        pygame.init()
        pygame.display.init()
        self.display = pygame.display.set_mode((self.window, self.window))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Mars Space Exploration')
        self.grid_size = int(self.window/self.size)
        self.grid_colour = (200, 200, 200)
        self.line_colour = (0, 0, 0)
        _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery = self.load_images()

        display_on = True

        reward = 0
        
        while display_on:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.display.fill(self.grid_colour)
            for x in range(0, self.window, self.grid_size):
                pygame.draw.line(self.display, self.line_colour, (x, 0), (x, self.window))
            for y in range(0, self.window, self.grid_size):
                pygame.draw.line(self.display, self.line_colour, (0, y), (self.window, y))

            self.render_images(_robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery)

            pygame.display.flip()

            if self.robot.position in self.cliffs:
                print("Robot fell on a cliff")
                time.sleep(2)
                pygame.quit()
                sys.exit()

            if self.robot.battery <= 0:
                print("Robot is out of battery")
                time.sleep(2)
                pygame.quit()
                sys.exit()

            print("--------------------------------")
            print(f"Battery: {self.robot.battery}")
            print(f"Reward: {reward}")
            print(f"Epsilon: {self.epsilon}")

            old_position = self.robot.position

            action = self.choose_action()
            
            self.update_robot(action)

            reward = self.calculate_reward(old_position)

            self.update_q_table(action, reward, old_position)

            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.decay_rate

            print("--------------------------------")

            time.sleep(3)

    def choose_action(self):
        possible_actions = []
            
        if self.robot.position[0] > 0:
            possible_actions.append(Actions.LEFT)
        if self.robot.position[0] < self.size-1:
            possible_actions.append(Actions.RIGHT)
        if self.robot.position[1] > 0:
            possible_actions.append(Actions.UP)
        if self.robot.position[1] < self.size-1:
            possible_actions.append(Actions.DOWN)

        if self.robot.holding_rock and self.robot.position in self.transmiter_stations:
            possible_actions.append(Actions.TRANSMIT)

        if self.robot.battery < 100 and self.robot.position in self.batery_stations:
            possible_actions.append(Actions.RECHARGE)

        if self.robot.position in self.rocks and not self.robot.holding_rock:
            possible_actions.append(Actions.COLLECT)

        print(f"Possible actions: {possible_actions}")

        if random.random() < self.epsilon:
            action = random.choice(possible_actions)
        else:
            best_action = None
            robot_position = self.robot.position[0] + self.size*self.robot.position[1]
            for action in possible_actions:
                if best_action is None or self.q_table[robot_position][action.value] > self.q_table[robot_position][best_action.value]:
                    best_action = action
            action = best_action

        return action

    def update_robot(self, action):
        if action == Actions.RIGHT:
            self.robot.position = (self.robot.position[0] + 1, self.robot.position[1])
            self.robot.battery -= 5
        elif action == Actions.LEFT:
            self.robot.position = (self.robot.position[0] - 1, self.robot.position[1])
            self.robot.battery -= 5
        elif action == Actions.UP:
            self.robot.position = (self.robot.position[0], self.robot.position[1] - 1)
            self.robot.battery -= 5
        elif action == Actions.DOWN:
            self.robot.position = (self.robot.position[0], self.robot.position[1] + 1)
            self.robot.battery -= 5
        elif action == Actions.COLLECT:
            self.robot.holding_rock = True
            self.rocks.remove(self.robot.position)
        elif action == Actions.RECHARGE:
            self.robot.battery = 100
        elif action == Actions.TRANSMIT:
            self.robot.holding_rock = False

        if self.robot.position in self.uphills:
            self.robot.battery -= 5
        if self.robot.position in self.downhills:
            self.robot.battery += 5

    def calculate_reward(self, old_position):
        reward = 0
        if self.robot.battery <= 0:
            reward -= 100

        reward += self.rewards[old_position[0] + self.size*old_position[1]][self.robot.position[0] + self.size*self.robot.position[1]]
        return reward

    def update_q_table(self, action, reward, old_position):
        old_q_value = self.q_table[old_position[0] + self.size*old_position[1]][action.value]
        max_future_q_value = np.max(self.q_table[self.robot.position[0] + self.size*self.robot.position[1]])
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q_value - old_q_value)
        self.q_table[old_position[0] + self.size*old_position[1]][action.value] = new_q_value

mars_environment = Mars_Environment(5)
