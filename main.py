import pygame
from enum import Enum
import sys
import os
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from game_window import GameWindow

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
        self.holding_rock_count = 0
        self.action = Actions.RIGHT

class Mars_Environment():
    def __init__(
        self, size,
        max_epsilon = 1,
        min_epsilon = 0.05,
        epsilon_decay_rate = 0.0001,
        alpha = 0.1,
        gamma = 0.9,
        no_episodes = 1000,
        max_steps = 100,
        policy = "episilon_greedy",
        max_temperature = 10,
        min_temperature = 0.1,
        temperature_decay_rate = 0.0001
    ):
        # Initialize game window
        self.game_window = GameWindow(800)
        self.size = size
        self.window = 800
        self.display = None
        self.clock = None
        
        self.max_epsilon = max_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.no_episodes = no_episodes
        self.max_steps = max_steps
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.policy = policy
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature_decay_rate = temperature_decay_rate

        self.q_table = []
        self.rewards = []
        self.robot = Robot()
        self.rocks = [(1,2), (3,3), (2,4)]
        self.transmiter_stations = [(4,4)]
        self.cliffs = [(2,3), (1,1)]
        self.uphills = [(0,4), (2,0)]
        self.downhills = [(3,0), (0,2)]
        self.batery_stations = [(4,2)]
        self.initialize_q_table()
        self.fill_rewards()
        self.render()

    def initialize_q_table(self):
        self.q_table = [[0 for _ in range(len(Actions))] for _ in range(self.size * self.size)]

    def reset(self):
        self.robot.position = (0, 0)
        self.robot.action = Actions.RIGHT
        self.robot.battery = 100
        self.robot.holding_rock_count = 0
        self.rocks = [(1,2), (3,3), (2,4)] 


    def fill_rewards(self):
        self.rewards = [[-5 for _ in range(self.size * self.size)] for _ in range(self.size * self.size)]

        for rock in self.rocks:
            self.rewards[rock[0] + self.size*rock[1]][rock[0] + self.size*rock[1]] += 150
        
        for transmiter in self.transmiter_stations:
            self.rewards[transmiter[0] + self.size*transmiter[1]][transmiter[0] + self.size*transmiter[1]] += 250
           
        for battery in self.batery_stations:
            self.rewards[battery[0] + self.size*battery[1]][battery[0] + self.size*battery[1]] += 100
          
        for cliff in self.cliffs:
            if cliff[0] > 0:
                self.rewards[cliff[0]-1 + self.size*cliff[1]][cliff[0] + self.size*cliff[1]] += -200
            if cliff[0] < self.size-1:
                self.rewards[cliff[0]+1 + self.size*cliff[1]][cliff[0] + self.size*cliff[1]] += -200
            if cliff[1] > 0:
                self.rewards[cliff[0] + self.size*(cliff[1]-1)][cliff[0] + self.size*cliff[1]] += -200
            if cliff[1] < self.size-1:
                self.rewards[cliff[0] + self.size*(cliff[1]+1)][cliff[0] + self.size*cliff[1]] += -200
        
        for uphill in self.uphills:
            if uphill[0] > 0:
                self.rewards[uphill[0]-1 + self.size*uphill[1]][uphill[0] + self.size*uphill[1]] += -10
            if uphill[0] < self.size-1:
                self.rewards[uphill[0]+1 + self.size*uphill[1]][uphill[0] + self.size*uphill[1]] += -10
            if uphill[1] > 0:
                self.rewards[uphill[0] + self.size*(uphill[1]-1)][uphill[0] + self.size*uphill[1]] += -10                
            if uphill[1] < self.size-1:
                self.rewards[uphill[0] + self.size*(uphill[1]+1)][uphill[0] + self.size*uphill[1]] += -10

        for downhill in self.downhills:
            if downhill[0] > 0:
                self.rewards[downhill[0]-1 + self.size*downhill[1]][downhill[0] + self.size*downhill[1]] += 1
            if downhill[0] < self.size-1:
                self.rewards[downhill[0]+1 + self.size*downhill[1]][downhill[0] + self.size*downhill[1]] += 1
            if downhill[1] > 0:
                self.rewards[downhill[0] + self.size*(downhill[1]-1)][downhill[0] + self.size*downhill[1]] += 1
            if downhill[1] < self.size-1:
                self.rewards[downhill[0] + self.size*(downhill[1]+1)][downhill[0] + self.size*downhill[1]] += 1

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
            pos = (self.game_window.sidebar_width + obj[0] * self.grid_size + self.grid_size * offset, 
                  obj[1] * self.grid_size + self.grid_size * offset)
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
        # Show start screen before beginning simulation
        if not self.game_window.show_start_screen():
            pygame.quit()
            sys.exit()
            
        pygame.init()
        pygame.display.init()
        self.display = self.game_window.display
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Mars Space Exploration')
        self.grid_size = int(self.window/self.size)
        self.grid_colour = (200, 200, 200)
        self.line_colour = (0, 0, 0)
        _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery = self.load_images()

        reward = 0
        steps = []
        episode_numbers = []
        
        # Initialize counters
        goal_reached_count = 0
        cliff_falls_count = 0
        battery_depleted_count = 0
        successful_mission_steps = []  # List to store steps for successful missions

        for episode in range(self.no_episodes):
            print(f"EPISODE NO: {episode+1}")
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.epsilon_decay_rate*episode)
            temperature = self.min_temperature + (self.max_temperature - self.min_temperature)*np.exp(-self.temperature_decay_rate*episode)

            self.reset()

            for step in range(self.max_steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Fill only the game area, not the sidebar
                pygame.draw.rect(self.display, self.grid_colour, 
                               (self.game_window.sidebar_width, 0, self.window, self.window))
                
                # Draw grid lines offset by sidebar width
                for x in range(self.game_window.sidebar_width, self.window + self.game_window.sidebar_width, self.grid_size):
                    pygame.draw.line(self.display, self.line_colour, (x, 0), (x, self.window))
                for y in range(0, self.window, self.grid_size):
                    pygame.draw.line(self.display, self.line_colour, 
                                   (self.game_window.sidebar_width, y), 
                                   (self.window + self.game_window.sidebar_width, y))

                # Update sidebar with additional information
                self.game_window.draw_sidebar(episode + 1, step + 1, epsilon, temperature, self.policy)

                # Adjust object rendering to account for sidebar offset
                self.render_images(_robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery)

                pygame.display.flip()


                if self.robot.position in self.cliffs:
                    print("--------------------------------")
                    print(f"STEP NO: {step+1} \n")
                    print(f"Position: {self.robot.position}")
                    print(f"Battery: {self.robot.battery}")
                    print(f"Reward: {reward}")
                    print(f"Epsilon: {epsilon}")
                    print(f"Temperature: {temperature}")
                    print("--------------------------------\n")

                    print("fell off a cliff")
                    cliff_falls_count += 1
                    # time.sleep(0.5)
                    
                    break

                if self.robot.battery <= 0:
                    print("--------------------------------")
                    print(f"STEP NO: {step+1} \n")
                    print(f"Position: {self.robot.position}")
                    print(f"Battery: {self.robot.battery}")
                    print(f"Reward: {reward}")
                    print(f"Epsilon: {epsilon}")
                    print(f"Temperature: {temperature}")
                    print("--------------------------------")
                    
                    print("ran out of battery")
                    battery_depleted_count += 1
                    # time.sleep(0.5)

                    break

                if self.robot.action == Actions.TRANSMIT:
                    print("--------------------------------")
                    print(f"STEP NO: {step+1} \n")
                    print(f"Position: {self.robot.position}")
                    print(f"Battery: {self.robot.battery}")
                    print(f"Reward: {reward}")
                    print(f"Epsilon: {epsilon}")
                    print(f"Temperature: {temperature}")
                    print("--------------------------------")

                    if not len(self.rocks):
                        print(f"Goal Reached, all rocks collected! Episode {episode+1} has ended")
                        goal_reached_count += 1
                        successful_mission_steps.append(step + 1)  # Add steps for successful mission
                        episode_numbers.append(episode)
                        steps.append(step)

                        # time.sleep(0.5)


                        break

                print("--------------------------------")
                print(f"STEP NO: {step+1} \n")
                print(f"Position: {self.robot.position}")
                print(f"Battery: {self.robot.battery}")
                print(f"Reward: {reward}")
                print(f"Epsilon: {epsilon}")
                print(f"Temperature: {temperature}")

                old_position = self.robot.position

                action = self.choose_action(epsilon, temperature)
                
                self.update_robot(action)

                reward = self.calculate_reward(old_position)

                self.update_q_table(action, reward, old_position)

                print("--------------------------------")

                # time.sleep(0.5)
        
        # Calculate averages
        avg_steps_successful_mission = sum(successful_mission_steps) / len(successful_mission_steps) if successful_mission_steps else 0
        
        # Display final statistics
        print("\n=== Final Statistics ===")
        print(f"Total Episodes: {self.no_episodes}")
        print(f"Goals Reached: {goal_reached_count} ({(goal_reached_count/self.no_episodes)*100:.2f}%)")
        print(f"Cliff Falls: {cliff_falls_count} ({(cliff_falls_count/self.no_episodes)*100:.2f}%)")
        print(f"Battery Depletions: {battery_depleted_count} ({(battery_depleted_count/self.no_episodes)*100:.2f}%)")
        print(f"Failed Episodes: {self.no_episodes - goal_reached_count - cliff_falls_count - battery_depleted_count} ({((self.no_episodes - goal_reached_count - cliff_falls_count - battery_depleted_count)/self.no_episodes)*100:.2f}%)")
        print(f"Average Steps for Successful Missions: {avg_steps_successful_mission:.2f}")
        print("=====================\n")
        
        bars = plt.bar(episode_numbers, steps, width=1.1)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        plt.xlabel('Episode')
        plt.ylabel('Steps') 
        plt.title('Steps per Episode')
        plt.grid(True)
        plt.show()


    def choose_action(self, epsilon, temperature):
        possible_actions = []
            
        if self.robot.position[0] > 0:
            possible_actions.append(Actions.LEFT)
        if self.robot.position[0] < self.size-1:
            possible_actions.append(Actions.RIGHT)
        if self.robot.position[1] > 0:
            possible_actions.append(Actions.UP)
        if self.robot.position[1] < self.size-1:
            possible_actions.append(Actions.DOWN)

        if self.robot.holding_rock_count > 0 and self.robot.position in self.transmiter_stations:
            possible_actions.append(Actions.TRANSMIT)

        if self.robot.battery < 50 and self.robot.position in self.batery_stations:
            possible_actions.append(Actions.RECHARGE)

        if self.robot.position in self.rocks:
            possible_actions.append(Actions.COLLECT)

        print(f"Possible actions: {possible_actions}")

        robot_position = self.robot.position[0] + self.size * self.robot.position[1]
        
        if self.policy == "episilon_greedy":
            if random.random() < epsilon:
                action = random.choice(possible_actions)
            else:
                best_action = None
                robot_position = self.robot.position[0] + self.size*self.robot.position[1]
                for action in possible_actions:
                    if best_action is None or self.q_table[robot_position][action.value] > self.q_table[robot_position][best_action.value]:
                        best_action = action
                action = best_action
        elif self.policy == "softmax":
            q_values = [self.q_table[robot_position][action.value] for action in possible_actions]
            exp_q = np.exp(np.array(q_values) / temperature)
            probabilities = exp_q / np.sum(exp_q)
            action = np.random.choice(possible_actions, p=probabilities)

        return action

    def update_robot(self, action):
        if action == Actions.RIGHT:
            self.robot.position = (self.robot.position[0] + 1, self.robot.position[1])
            self.robot.battery -= 2
        elif action == Actions.LEFT:
            self.robot.position = (self.robot.position[0] - 1, self.robot.position[1])
            self.robot.battery -= 2
        elif action == Actions.UP:
            self.robot.position = (self.robot.position[0], self.robot.position[1] - 1)
            self.robot.battery -= 2
        elif action == Actions.DOWN:
            self.robot.position = (self.robot.position[0], self.robot.position[1] + 1)
            self.robot.battery -= 2
        elif action == Actions.COLLECT:
            self.robot.holding_rock_count += 1
            self.rocks.remove(self.robot.position)
        elif action == Actions.RECHARGE:
            self.robot.battery = 100
        elif action == Actions.TRANSMIT:
            self.robot.holding_rock_count -= 1
            

        if self.robot.position in self.uphills:
            self.robot.battery -= 2
        if self.robot.position in self.downhills:
            self.robot.battery += 2

        self.robot.action = action

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

mars_environment = Mars_Environment(
    size=5,
    max_epsilon = 1,
    min_epsilon = 0.05,
    epsilon_decay_rate = 0.0005,
    alpha = 0.7,
    gamma = 0.4,
    no_episodes = 2000,
    max_steps = 75,
    policy="softmax",
    max_temperature=100,
    min_temperature=0.01,
    temperature_decay_rate=0.0005
)
