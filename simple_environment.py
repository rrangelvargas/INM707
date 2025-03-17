import pygame
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from game_window import GameWindow
from utils import Actions, Robot

class Mars_Environment():
    def __init__(
        self, 
        config,
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
        self.config = config
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
        self.size = self.config["size"]
        self.rocks = self.config["rocks"]
        self.transmiter_stations = self.config["transmiter_stations"]
        self.cliffs = self.config["cliffs"]
        self.uphills = self.config["uphills"]
        self.downhills = self.config["downhills"]
        self.batery_stations = self.config["battery_stations"]
        self.initialize_q_table()
        self.fill_rewards()
        self.game_window = GameWindow(800)

    def initialize_q_table(self):
        self.q_table = [[0 for _ in range(len(Actions))] for _ in range(self.size * self.size)]

    def reset(self):
        self.robot.position = (0, 0)
        self.robot.battery = 100
        self.robot.holding_rock_count = 0
        self.rocks = self.config["rocks"]

    def fill_rewards(self):
        self.rewards = [[-5 for _ in range(self.size * self.size)] for _ in range(self.size * self.size)]

        for rock in self.rocks:
            self.rewards[rock[0] + self.size * rock[1]][rock[0] + self.size * rock[1]] += 150
        
        for transmiter in self.transmiter_stations:
            self.rewards[transmiter[0] + self.size * transmiter[1]][transmiter[0] + self.size * transmiter[1]] += 250
           
        for battery in self.batery_stations:
            self.rewards[battery[0] + self.size * battery[1]][battery[0] + self.size * battery[1]] += 100
          
        for cliff in self.cliffs:
            if cliff[0] > 0:
                self.rewards[cliff[0]-1 + self.size * cliff[1]][cliff[0] + self.size * cliff[1]] += -200
            if cliff[0] < self.size-1:
                self.rewards[cliff[0]+1 + self.size * cliff[1]][cliff[0] + self.size * cliff[1]] += -200
            if cliff[1] > 0:
                self.rewards[cliff[0] + self.size * (cliff[1]-1)][cliff[0] + self.size * cliff[1]] += -200
            if cliff[1] < self.size-1:
                self.rewards[cliff[0] + self.size * (cliff[1]+1)][cliff[0] + self.size * cliff[1]] += -200
        
        for uphill in self.uphills:
            if uphill[0] > 0:
                self.rewards[uphill[0]-1 + self.size * uphill[1]][uphill[0] + self.size * uphill[1]] += -10
            if uphill[0] < self.size-1:
                self.rewards[uphill[0]+1 + self.size * uphill[1]][uphill[0] + self.size * uphill[1]] += -10
            if uphill[1] > 0:
                self.rewards[uphill[0] + self.size * (uphill[1]-1)][uphill[0] + self.size * uphill[1]] += -10                
            if uphill[1] < self.size-1:
                self.rewards[uphill[0] + self.size * (uphill[1]+1)][uphill[0] + self.size * uphill[1]] += -10

        for downhill in self.downhills:
            if downhill[0] > 0:
                self.rewards[downhill[0]-1 + self.size * downhill[1]][downhill[0] + self.size * downhill[1]] += 1
            if downhill[0] < self.size-1:
                self.rewards[downhill[0]+1 + self.size * downhill[1]][downhill[0] + self.size * downhill[1]] += 1
            if downhill[1] > 0:
                self.rewards[downhill[0] + self.size * (downhill[1]-1)][downhill[0] + self.size * downhill[1]] += 1
            if downhill[1] < self.size-1:
                self.rewards[downhill[0] + self.size * (downhill[1]+1)][downhill[0] + self.size * downhill[1]] += 1

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
        robot_index = self.robot.position[0] + self.size * self.robot.position[1]
        
        if self.policy == "episilon_greedy":
            if random.random() < epsilon:
                action = random.choice(possible_actions)
            else:
                best_action = None
                for act in possible_actions:
                    if best_action is None or self.q_table[robot_index][act.value] > self.q_table[robot_index][best_action.value]:
                        best_action = act
                action = best_action
        elif self.policy == "softmax":
            q_values = [self.q_table[robot_index][act.value] for act in possible_actions]
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
            if self.robot.position in self.rocks:
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
        new_index = self.robot.position[0] + self.size * self.robot.position[1]
        old_index = old_position[0] + self.size * old_position[1]
        reward += self.rewards[old_index][new_index]
        return reward

    def update_q_table(self, action, reward, old_position):
        pos_index = old_position[0] + self.size * old_position[1]
        old_q_value = self.q_table[pos_index][action.value]
        new_state_index = self.robot.position[0] + self.size * self.robot.position[1]
        max_future_q = np.max(self.q_table[new_state_index])
        new_q = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        self.q_table[pos_index][action.value] = new_q

    def run(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption('Mars Space Exploration')
        grid_size = int(self.game_window.window_size / self.size)

        reward = 0
        goal_reached_count = 0
        cliff_falls_count = 0
        battery_depleted_count = 0
        successful_mission_steps = []
        steps = []
        episode_numbers = []

        for episode in range(self.no_episodes):
            print(f"EPISODE NO: {episode+1}")
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)
            temperature = self.min_temperature + (self.max_temperature - self.min_temperature) * np.exp(-self.temperature_decay_rate * episode)
            self.reset()

            for step in range(self.max_steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                pygame.draw.rect(self.game_window.display, self.game_window.GRID_COLOR, 
                                 (self.game_window.sidebar_width, 0, self.game_window.window_size, self.game_window.window_size))
                
                self.game_window.draw_grid(grid_size)

                self.game_window.draw_sidebar(episode + 1, step + 1, epsilon, temperature, self.policy)

                self.game_window.render_images(self.robot, self.rocks, self.transmiter_stations,
                                                 self.cliffs, self.uphills, self.downhills,
                                                 self.batery_stations, grid_size)
                pygame.display.flip()


                if self.robot.position in self.cliffs:
                    print("--------------------------------")
                    print(f"STEP {step+1}: Position {self.robot.position}, Battery {self.robot.battery}, Reward {reward}")
                    print("Fell off a cliff")
                    cliff_falls_count += 1
                    break

                if self.robot.battery <= 0:
                    print("--------------------------------")
                    print(f"STEP {step+1}: Position {self.robot.position}, Battery {self.robot.battery}, Reward {reward}")
                    print("Battery depleted")
                    battery_depleted_count += 1
                    break

                if self.robot.action == Actions.TRANSMIT:
                    print("--------------------------------")
                    print(f"STEP {step+1}: Position {self.robot.position}, Battery {self.robot.battery}, Reward {reward}")
                    if not self.rocks:
                        print(f"Goal reached, all rocks collected! Episode {episode+1} ended.")
                        goal_reached_count += 1
                        successful_mission_steps.append(step + 1)
                        episode_numbers.append(episode)
                        steps.append(step)
                        break

                print("--------------------------------")
                print(f"STEP {step+1}: Position {self.robot.position}, Battery {self.robot.battery}, Reward {reward}")
                print(f"Epsilon: {epsilon:.4f} Temperature: {temperature:.4f}")

                old_position = self.robot.position
                action = self.choose_action(epsilon, temperature)
                self.update_robot(action)
                reward = self.calculate_reward(old_position)
                self.update_q_table(action, reward, old_position)
                print("--------------------------------")

        avg_steps = sum(successful_mission_steps) / len(successful_mission_steps) if successful_mission_steps else 0
        print("\n=== Final Statistics ===")
        print(f"Total Episodes: {self.no_episodes}")
        print(f"Goals Reached: {goal_reached_count} ({(goal_reached_count/self.no_episodes)*100:.2f}%)")
        print(f"Cliff Falls: {cliff_falls_count} ({(cliff_falls_count/self.no_episodes)*100:.2f}%)")
        print(f"Battery Depletions: {battery_depleted_count} ({(battery_depleted_count/self.no_episodes)*100:.2f}%)")
        failed = self.no_episodes - goal_reached_count - cliff_falls_count - battery_depleted_count
        print(f"Failed Episodes: {failed} ({(failed/self.no_episodes)*100:.2f}%)")
        print(f"Average Steps for Successful Missions: {avg_steps:.2f}")
        print("=====================\n")
        
        if episode_numbers and steps:
            bars = plt.bar(episode_numbers, steps, width=1.1)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
            plt.xlabel('Episode')
            plt.ylabel('Steps') 
            plt.title('Steps per Episode')
            plt.grid(True)
            plt.show()
