import torch
from game_window import GameWindow
from utils import Actions, Robot, device, Transition, Entities
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pygame
import sys
import time


def convert_state(state, size):
    
    c = np.array(state["position"])/size
    o = state["surrounding"].flatten()/size
    state_tensor = np.concatenate([c,o])
    state_tensor = torch.tensor(state_tensor, device=device).unsqueeze(0)
    
    return state_tensor
    
class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, size):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        state_tensor = convert_state(state, size)

        if next_state is None:
            state_tensor_next = None            
        else:
            state_tensor_next = convert_state(next_state, size)
        
        action_tensor = torch.tensor([action], device=device).unsqueeze(0)

        reward = torch.tensor([reward], device=device).unsqueeze(0)/10

        self.memory[self.position] = Transition(state_tensor, action_tensor, state_tensor_next, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)   
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        h1 = F.relu(self.bn1(self.fc1(x.float())))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h3 = F.relu(self.bn3(self.fc3(h2)))
        output = self.fc4(h3.view(h3.size(0), -1))
        return output


class E_Greedy_Policy():
    def __init__(self, epsilon, decay, min_epsilon):
        
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.decay = decay
        self.epsilon_min = min_epsilon
                
    def __call__(self, state, network):
                
        is_greedy = random.random() > self.epsilon
        
        if is_greedy:
            # we select greedy action
            with torch.no_grad():
                network.eval()
                # index of the maximum over dimension 1.
                index_action = network(state).max(1)[1].view(1, 1).cpu()[0][0].item()
                
                network.train()
        else:
            # we sample a random action
            index_action = random.randint(0,3)
        
        return index_action
                
    def update_epsilon(self):
        
        self.epsilon = self.epsilon*self.decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        
    def reset(self):
        self.epsilon = self.epsilon_start

class Deep_Q_Mars():
    def __init__(self,config, no_episodes = 1000, max_steps = 100, epsilon = 0.99, decay = 0.997, min_epsilon = 0.001):
        print(f"Using device: {device}")
        
        self.config = config
        self.game_window = GameWindow(800)
        self.size = self.config["size"]
        self.no_episodes = no_episodes
        self.max_steps = max_steps

        input_size = 11
        hidden_size = 128
        output_size = len(Actions)

        self.network = DQN(input_size, hidden_size, output_size).to(device)
        self.target_network = DQN(input_size, hidden_size, output_size).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01)
        self.memory = ReplayMemory(10000)
        self.policy = E_Greedy_Policy(epsilon, decay, min_epsilon)

        self.robot = Robot()
        self.rocks = self.config["rocks"]
        self.uphills = self.config["uphills"]
        self.downhills = self.config["downhills"]
        self.transmiter_stations = self.config["transmiter_stations"]
        self.battery_stations = self.config["battery_stations"]
        self.cliffs = self.config["cliffs"]

        self.rewards_history = []

    def warm_up(self, state):
        print("Warm up started")
        memory_filled = False
        total_reward = 0
        while not memory_filled:
            state_tensor = convert_state(state, self.size)

            action_index = self.policy(state_tensor, self.network)
            action = Actions(action_index)

            next_state, reward, done = self.step(action)

            total_reward += float(reward)

            if done:
                next_state = None

            self.memory.push(state, action_index, next_state, reward, self.size)

            memory_filled = len(self.memory) == self.memory.capacity

        print(f"Warm up completed, total reward: {total_reward}")

    def run(self):

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption('Deep Q Mars Space Exploration')
        grid_size = int(self.game_window.window_size / self.size)

        pygame.draw.rect(self.game_window.display, self.game_window.GRID_COLOR, 
                                 (self.game_window.sidebar_width, 0, self.game_window.window_size, self.game_window.window_size))
        
        self.game_window.draw_grid(grid_size)


        self.policy.reset()

        self.warm_up(self.reset())
        
        for episode in range(self.no_episodes):
            print(f"EPISODE NO: {episode+1}")
            state = self.reset()
            done = False
            total_reward = 0

            for step in range(self.max_steps):
                self.game_window.draw_sidebar(episode + 1, step + 1)

                self.game_window.render_images(self.robot, self.rocks, self.transmiter_stations,
                                                    self.cliffs, self.uphills, self.downhills,
                                                    self.battery_stations, grid_size)
            
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                if done:
                    next_state = None

                state_tensor = convert_state(state, self.size)

                action_index = self.policy(state_tensor, self.network)
                action = Actions(action_index)

                next_state, reward, done = self.step(action)

                total_reward += float(reward)

                self.memory.push(state, action_index, next_state, reward, self.size)

                state = next_state
            
            self.policy.update_epsilon()
            self.rewards_history.append(total_reward)
            pygame.display.flip()

            time.sleep(1)

    
    def reset(self):
        self.robot.position = (0, 0)
        self.robot.battery = 100
        self.robot.holding_rock_count = 0
        self.rocks = self.config["rocks"]

        return {"position": self.robot.position, "surrounding": self.calculate_surrounding()}

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

        if self.robot.holding_rock_count > 0 and self.robot.position in self.transmiter_stations:
            possible_actions.append(Actions.TRANSMIT)

        if self.robot.battery < 50 and self.robot.position in self.battery_stations:
            possible_actions.append(Actions.RECHARGE)

        if self.robot.position in self.rocks:
            possible_actions.append(Actions.COLLECT)

        print(f"Possible actions: {possible_actions}")

        state = {"position": self.robot.position, "surrounding": self.calculate_surrounding()}

        state_tensor = convert_state(state, self.size)

        action = self.network.forward(state_tensor)

        return action
    
    def step(self, action):
        old_position = self.robot.position
        self.update_robot(action)
        next_state = {"position": self.robot.position, "surrounding": self.calculate_surrounding()}
        reward = self.calculate_reward(old_position)
        done = self.robot.battery <= 0 or self.robot.position in self.cliffs
        return next_state, reward, done
    
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
        reward = -5

        if self.robot.position in self.rocks and self.robot.position == old_position:
            reward += 150
        elif self.robot.position in self.uphills:
            reward += -10
        elif self.robot.position in self.downhills:
            reward += 1
        elif self.robot.position in self.transmiter_stations and self.robot.position == old_position:
            reward += 250
        elif self.robot.position in self.battery_stations and self.robot.position == old_position:
            reward += 100
        elif self.robot.position in self.cliffs:
            reward += -200

        return reward
    

    def calculate_surrounding(self):
        position = self.robot.position

        obs = np.zeros((3, 3), dtype = np.int8)

        if position[0] == 0:
            obs[0, :] = Entities.EDGE.value
        if position[0] == self.size - 1:
            obs[2, :] = Entities.EDGE.value
        if position[1] == 0:
            obs[:, 0] = Entities.EDGE.value
        if position[1] == self.size - 1:
            obs[:, 2] = Entities.EDGE.value

        for cliff in self.cliffs:
            if (position[0] - 1, position[1] - 1) == cliff:
                obs[0, 0] = Entities.CLIFF.value
            if (position[0] - 1, position[1]) == cliff:
                obs[0, 1] = Entities.CLIFF.value
            if (position[0] - 1, position[1] + 1) == cliff:
                obs[0, 2] = Entities.CLIFF.value
            if (position[0], position[1] - 1) == cliff:
                obs[1, 0] = Entities.CLIFF.value        
            if (position[0], position[1] + 1) == cliff: 
                obs[1, 2] = Entities.CLIFF.value
            if (position[0] + 1, position[1] - 1) == cliff:
                obs[2, 0] = Entities.CLIFF.value
            if (position[0] + 1, position[1]) == cliff:
                obs[2, 1] = Entities.CLIFF.value
            if (position[0] + 1, position[1] + 1) == cliff:
                obs[2, 2] = Entities.CLIFF.value

        for rock in self.rocks:
            if (position[0] - 1, position[1] - 1) == rock:
                obs[0, 0] = Entities.ROCK.value
            if (position[0] - 1, position[1]) == rock:
                obs[0, 1] = Entities.ROCK.value
            if (position[0] - 1, position[1] + 1) == rock:
                obs[0, 2] = Entities.ROCK.value
            if (position[0], position[1] - 1) == rock:
                obs[1, 0] = Entities.ROCK.value
            if (position[0], position[1] + 1) == rock:
                obs[1, 2] = Entities.ROCK.value
            if (position[0] + 1, position[1] - 1) == rock:
                obs[2, 0] = Entities.ROCK.value
            if (position[0] + 1, position[1]) == rock:
                obs[2, 1] = Entities.ROCK.value
            if (position[0] + 1, position[1] + 1) == rock:
                obs[2, 2] = Entities.ROCK.value

        for uphill in self.uphills:
            if (position[0] - 1, position[1] - 1) == uphill:
                obs[0, 0] = Entities.UPHILL.value
            if (position[0] - 1, position[1]) == uphill:
                obs[0, 1] = Entities.UPHILL.value
            if (position[0] - 1, position[1] + 1) == uphill:
                obs[0, 2] = Entities.UPHILL.value   
            if (position[0], position[1] - 1) == uphill:    
                obs[1, 0] = Entities.UPHILL.value
            if (position[0], position[1] + 1) == uphill:
                obs[1, 2] = Entities.UPHILL.value
            if (position[0] + 1, position[1] - 1) == uphill:
                obs[2, 0] = Entities.UPHILL.value   
            if (position[0] + 1, position[1]) == uphill:    
                obs[2, 1] = Entities.UPHILL.value
            if (position[0] + 1, position[1] + 1) == uphill:
                obs[2, 2] = Entities.UPHILL.value

        for downhill in self.downhills: 
            if (position[0] - 1, position[1] - 1) == downhill:
                obs[0, 0] = Entities.DOWNHILL.value
            if (position[0] - 1, position[1]) == downhill:
                obs[0, 1] = Entities.DOWNHILL.value
            if (position[0] - 1, position[1] + 1) == downhill:
                obs[0, 2] = Entities.DOWNHILL.value         
            if (position[0], position[1] - 1) == downhill:
                obs[1, 0] = Entities.DOWNHILL.value
            if (position[0], position[1] + 1) == downhill:
                obs[1, 2] = Entities.DOWNHILL.value
            if (position[0] + 1, position[1] - 1) == downhill:
                obs[2, 0] = Entities.DOWNHILL.value
            if (position[0] + 1, position[1]) == downhill:
                obs[2, 1] = Entities.DOWNHILL.value
            if (position[0] + 1, position[1] + 1) == downhill:
                obs[2, 2] = Entities.DOWNHILL.value

        for transmiter_station in self.transmiter_stations:
            if (position[0] - 1, position[1] - 1) == transmiter_station:
                obs[0, 0] = Entities.TRANSMITER_STATION.value
            if (position[0] - 1, position[1]) == transmiter_station:
                obs[0, 1] = Entities.TRANSMITER_STATION.value
            if (position[0] - 1, position[1] + 1) == transmiter_station:
                obs[0, 2] = Entities.TRANSMITER_STATION.value
            if (position[0], position[1] - 1) == transmiter_station:
                obs[1, 0] = Entities.TRANSMITER_STATION.value
            if (position[0], position[1] + 1) == transmiter_station:
                obs[1, 2] = Entities.TRANSMITER_STATION.value
            if (position[0] + 1, position[1] - 1) == transmiter_station:
                obs[2, 0] = Entities.TRANSMITER_STATION.value
            if (position[0] + 1, position[1]) == transmiter_station:
                obs[2, 1] = Entities.TRANSMITER_STATION.value
            if (position[0] + 1, position[1] + 1) == transmiter_station:
                obs[2, 2] = Entities.TRANSMITER_STATION.value

        for battery_station in self.battery_stations:
            if (position[0] - 1, position[1] - 1) == battery_station:
                obs[0, 0] = Entities.BATTERY_STATION.value
            if (position[0] - 1, position[1]) == battery_station:
                obs[0, 1] = Entities.BATTERY_STATION.value
            if (position[0] - 1, position[1] + 1) == battery_station:
                obs[0, 2] = Entities.BATTERY_STATION.value
            if (position[0], position[1] - 1) == battery_station:
                obs[1, 0] = Entities.BATTERY_STATION.value
            if (position[0], position[1] + 1) == battery_station:
                obs[1, 2] = Entities.BATTERY_STATION.value
            if (position[0] + 1, position[1] - 1) == battery_station:
                obs[2, 0] = Entities.BATTERY_STATION.value
            if (position[0] + 1, position[1]) == battery_station:
                obs[2, 1] = Entities.BATTERY_STATION.value
            if (position[0] + 1, position[1] + 1) == battery_station:
                obs[2, 2] = Entities.BATTERY_STATION.value             
                
        obs[1, 1] = Entities.ROBOT.value
                
        return obs
    
