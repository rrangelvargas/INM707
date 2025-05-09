import pygame
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from game_window import GameWindow
from utils import Actions, Robot
import time

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
        temperature_decay_rate = 0.0001,
        save_results = False,
        display = False
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
        self.save_results = save_results
        self.display = display

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
        self.game_window = GameWindow(800) if self.display else None

    def initialize_q_table(self): # initialising a 2d q-table for the state and action pairs
        self.q_table = [[0 for _ in range(len(Actions))] for _ in range(self.size * self.size)]

    def reset(self): # is called to reset the robot after an episode has finished
        self.robot.position = [0, 0]
        self.robot.battery = 100
        self.robot.holding_rock_count = 0
        self.robot.action = Actions.RIGHT
        self.rocks = self.config["rocks"].copy()

    def fill_rewards(self): # creating the reward matrix based on the positionings of the different objects in the grid
        self.rewards = [[-10 for _ in range(self.size * self.size)] for _ in range(self.size * self.size)]

        for rock in self.rocks: # the reward for each rock is 150
            self.rewards[rock[0] + self.size * rock[1]][rock[0] + self.size * rock[1]] += 150
        
        for transmiter in self.transmiter_stations: # the reward for each transmiter is 250
            self.rewards[transmiter[0] + self.size * transmiter[1]][transmiter[0] + self.size * transmiter[1]] += 250
           
        for battery in self.batery_stations: # the reward for each battery station is 150
            self.rewards[battery[0] + self.size * battery[1]][battery[0] + self.size * battery[1]] += 150
          
        for cliff in self.cliffs: # the penalty for each cliff is -250
            if cliff[0] > 0:
                self.rewards[cliff[0]-1 + self.size * cliff[1]][cliff[0] + self.size * cliff[1]] += -250
            if cliff[0] < self.size-1:
                self.rewards[cliff[0]+1 + self.size * cliff[1]][cliff[0] + self.size * cliff[1]] += -250
            if cliff[1] > 0:
                self.rewards[cliff[0] + self.size * (cliff[1]-1)][cliff[0] + self.size * cliff[1]] += -250
            if cliff[1] < self.size-1:
                self.rewards[cliff[0] + self.size * (cliff[1]+1)][cliff[0] + self.size * cliff[1]] += -250
        
        for uphill in self.uphills: # the penalty for each uphill is -10
            if uphill[0] > 0:
                self.rewards[uphill[0]-1 + self.size * uphill[1]][uphill[0] + self.size * uphill[1]] += -10
            if uphill[0] < self.size-1:
                self.rewards[uphill[0]+1 + self.size * uphill[1]][uphill[0] + self.size * uphill[1]] += -10
            if uphill[1] > 0:
                self.rewards[uphill[0] + self.size * (uphill[1]-1)][uphill[0] + self.size * uphill[1]] += -10                
            if uphill[1] < self.size-1:
                self.rewards[uphill[0] + self.size * (uphill[1]+1)][uphill[0] + self.size * uphill[1]] += -10

        for downhill in self.downhills: # the reward for each downhill is 1
            if downhill[0] > 0:
                self.rewards[downhill[0]-1 + self.size * downhill[1]][downhill[0] + self.size * downhill[1]] += 1
            if downhill[0] < self.size-1:
                self.rewards[downhill[0]+1 + self.size * downhill[1]][downhill[0] + self.size * downhill[1]] += 1
            if downhill[1] > 0:
                self.rewards[downhill[0] + self.size * (downhill[1]-1)][downhill[0] + self.size * downhill[1]] += 1
            if downhill[1] < self.size-1:
                self.rewards[downhill[0] + self.size * (downhill[1]+1)][downhill[0] + self.size * downhill[1]] += 1

    def choose_action(self, epsilon, temperature): # chooses the action the robot takes based on the policy
        possible_actions = []
        if self.robot.position[0] > 0: # if move left is a possible action
            possible_actions.append(Actions.LEFT)
        if self.robot.position[0] < self.size-1: # if move right is a possible action
            possible_actions.append(Actions.RIGHT)
        if self.robot.position[1] > 0: # if move up is a possible action
            possible_actions.append(Actions.UP)
        if self.robot.position[1] < self.size-1: # if move down is a possible action
            possible_actions.append(Actions.DOWN)
        if self.robot.holding_rock_count > 0 and self.robot.position in self.transmiter_stations: # if transmit is a possible action
            possible_actions.append(Actions.TRANSMIT)
        if self.robot.battery < 100 and self.robot.position in self.batery_stations: # if recharge is a possible action
            possible_actions.append(Actions.RECHARGE)
        if self.robot.position in self.rocks: # if collect is a possible action
            possible_actions.append(Actions.COLLECT)

        robot_index = self.robot.position[0] + self.size * self.robot.position[1]
        
        if self.policy == "epsilon_greedy": # the epsilon greedy policy
            if random.random() < epsilon:
                action = random.choice(possible_actions) # choose a random action
            else:
                best_action = None
                for act in possible_actions:
                    if best_action is None or self.q_table[robot_index][act.value] > self.q_table[robot_index][best_action.value]: 
                        best_action = act
                action = best_action
        elif self.policy == "softmax": # the softmax policy
            q_values = [self.q_table[robot_index][act.value] for act in possible_actions] # gets the q values of the possible actions
            exp_q = np.exp(np.array(q_values) / temperature) # calculates the exponentiated q values with temperature
            probabilities = exp_q / np.sum(exp_q) # calculates the probabilities of each action
            action = np.random.choice(possible_actions, p=probabilities) # chooses an action based on the probabilities
        return action

    def update_robot(self, action): # updates the state of the robot based on the action that was taken
        if action == Actions.RIGHT: # if the action is to move right
            self.robot.position = [self.robot.position[0] + 1, self.robot.position[1]]
            self.robot.battery -= 2
        elif action == Actions.LEFT: # if the action is to move left
            self.robot.position = [self.robot.position[0] - 1, self.robot.position[1]]
            self.robot.battery -= 2
        elif action == Actions.UP: # if the action is to move up
            self.robot.position = [self.robot.position[0], self.robot.position[1] - 1]
            self.robot.battery -= 2
        elif action == Actions.DOWN: # if the action is to move down
            self.robot.position = [self.robot.position[0], self.robot.position[1] + 1]
            self.robot.battery -= 2
        elif action == Actions.COLLECT: # if the action is to collect
            self.robot.holding_rock_count += 1
            if self.robot.position in self.rocks:
                self.rocks.remove(self.robot.position)
        elif action == Actions.RECHARGE: # if the action is to recharge
            self.robot.battery = 100
        elif action == Actions.TRANSMIT: # if the action is to transmit
            self.robot.holding_rock_count -= 1
            
        if self.robot.position in self.uphills: # if the robot is on an uphill
            self.robot.battery -= 2
        if self.robot.position in self.downhills: # if the robot is on an downhill
            self.robot.battery += 2

        self.robot.action = action

    def calculate_reward(self, old_position): # returns the reward from moing from the old to the new position
        reward = 0
        new_index = self.robot.position[0] + self.size * self.robot.position[1]
        old_index = old_position[0] + self.size * old_position[1]
        reward += self.rewards[old_index][new_index]
        return reward

    def update_q_table(self, action, reward, old_position): # updates the q value of the state action pair
        pos_index = old_position[0] + self.size * old_position[1]
        old_q_value = self.q_table[pos_index][action.value]
        new_state_index = self.robot.position[0] + self.size * self.robot.position[1]
        max_future_q = np.max(self.q_table[new_state_index])
        new_q = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        self.q_table[pos_index][action.value] = new_q

    def run(self): # the main training loop for a set number of episodes
        if self.display:
            grid_size = int(self.game_window.window_size / self.size)
        reward = 0
        goal_reached_count = 0
        cliff_falls_count = 0
        battery_depleted_count = 0
        successful_mission_steps = []
        episode_numbers = []
        total_rewards = []

        for episode in range(self.no_episodes): # runs for a set number of episodes
            episode_reward = 0
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode) # updating epsilon
            temperature = self.min_temperature + (self.max_temperature - self.min_temperature) * np.exp(-self.temperature_decay_rate * episode) # updating temperature
            self.reset()

            if not self.display:
                print(f"EPISODE NO: {episode+1}")

            for step in range(self.max_steps):
                if self.display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                    pygame.draw.rect(self.game_window.display, self.game_window.GRID_COLOR, # draws the window
                                     (self.game_window.sidebar_width, 0, self.game_window.window_size, self.game_window.window_size))
                    
                    self.game_window.draw_grid(grid_size) # draws the grid based on the size set

                    self.game_window.draw_sidebar(episode + 1, step + 1, self.robot.battery, epsilon, temperature, self.policy) # draws the sidebar for visualising results in the gui

                    self.game_window.render_images(self.robot, self.rocks, self.transmiter_stations, # renders the images
                                                     self.cliffs, self.uphills, self.downhills,
                                                     self.batery_stations, grid_size)
                    pygame.display.flip()


                if self.robot.position in self.cliffs: # if the robot is on a cliff
                    cliff_falls_count += 1
                    break

                if self.robot.battery <= 0: # if the robot has no battery
                    battery_depleted_count += 1
                    break

                if self.robot.action == Actions.TRANSMIT: # checks if the robot has reached the goal
                    if not self.rocks:
                        goal_reached_count += 1
                        successful_mission_steps.append(step)
                        episode_numbers.append(episode)
                        break

                old_position = self.robot.position # saves the old position
                action = self.choose_action(epsilon, temperature) # chooses the action
                self.update_robot(action) # updates the robot based on this action
                reward = self.calculate_reward(old_position) # calculates the reward from moving from the old to the new position
                episode_reward += reward # adds the reward to the episode reward
                self.update_q_table(action, reward, old_position) # updates the q table
            
            total_rewards.append(episode_reward) # adds the episode reward to the total rewards
                # time.sleep(1)

        avg_steps = sum(successful_mission_steps) / len(successful_mission_steps) if successful_mission_steps else 0
        print("\n=== Final Statistics ===") # the final statistics are printed once the training has finished
        print(f"Total Episodes: {self.no_episodes}")
        print(f"Goals Reached: {goal_reached_count} ({(goal_reached_count/self.no_episodes)*100:.2f}%)")
        print(f"Cliff Falls: {cliff_falls_count} ({(cliff_falls_count/self.no_episodes)*100:.2f}%)")
        print(f"Battery Depletions: {battery_depleted_count} ({(battery_depleted_count/self.no_episodes)*100:.2f}%)")
        failed = self.no_episodes - goal_reached_count - cliff_falls_count - battery_depleted_count
        print(f"Failed Episodes: {failed} ({(failed/self.no_episodes)*100:.2f}%)")
        print(f"Average Steps for Successful Missions: {avg_steps:.2f}")
        print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")
        print("=====================\n")

        if self.save_results: # saving the results if it has been set
            self.save_results_files(
                episode_numbers,
                successful_mission_steps,
                goal_reached_count,
                cliff_falls_count,
                battery_depleted_count,
                failed,
                avg_steps,
                total_rewards
            )


    def save_results_files( # the function for saving the results
            self,
            episode_numbers,
            successful_mission_steps,
            goal_reached_count,
            cliff_falls_count,
            battery_depleted_count,
            failed,
            avg_steps,
            total_rewards
        ):       
        with open("results_5x5_simple_game.txt", "w") as f:
            f.write("=====================\n")
            f.write(f"Policy: {self.policy}\n")
            f.write(f"Epsilon: {self.max_epsilon}\n") if self.policy == "epsilon_greedy" else f.write(f"Temperature: {self.max_temperature}\n")
            f.write(f"Min Epsilon: {self.min_epsilon}\n") if self.policy == "epsilon_greedy" else f.write(f"Min Temperature: {self.min_temperature}\n")
            f.write(f"Epsilon Decay Rate: {self.epsilon_decay_rate}\n") if self.policy == "epsilon_greedy" else f.write(f"Temperature Decay Rate: {self.temperature_decay_rate}\n")
            f.write(f"Alpha: {self.alpha}\n")
            f.write(f"Gamma: {self.gamma}\n")
            f.write(f"No Episodes: {self.no_episodes}\n")
            f.write(f"Max Steps: {self.max_steps}\n")
            f.write("=====================\n\n\n")
            f.write("Results:\n\n")
            f.write(f"Total Episodes: {self.no_episodes}\n")
            f.write(f"Goals Reached: {goal_reached_count} ({(goal_reached_count/self.no_episodes)*100:.2f}%)\n")
            f.write(f"Cliff Falls: {cliff_falls_count} ({(cliff_falls_count/self.no_episodes)*100:.2f}%)\n")
            f.write(f"Battery Depletions: {battery_depleted_count} ({(battery_depleted_count/self.no_episodes)*100:.2f}%)\n")
            f.write(f"Failed Episodes: {failed} ({(failed/self.no_episodes)*100:.2f}%)\n")
            f.write(f"Average Steps for Successful Missions: {avg_steps:.2f}\n")
            f.write(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}\n")

        if episode_numbers and successful_mission_steps: # the matplot graph for success runs
            plt.bar(episode_numbers, successful_mission_steps, width=1.1)
            plt.xlabel('Episode')
            plt.ylabel('Steps') 
            plt.title(f'Steps per Episode for {self.policy} Policy\nalpha: {self.alpha} - gamma: {self.gamma}')
            plt.grid(True)
            plt.savefig(f"steps_per_episode_{self.policy}.png")

        if total_rewards: # the matplot graph for total rewards
            plt.figure(figsize=(10, 5))
            plt.plot(total_rewards, label='Total Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'Total Rewards for {self.policy} Policy\nalpha: {self.alpha} - gamma: {self.gamma}')
            plt.grid(True)
            plt.savefig(f"total_rewards_{self.policy}.png")
