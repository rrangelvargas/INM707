import torch
from game_window import GameWindow
from utils import Actions, Robot, device, Transition, Entities
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


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


class Deep_Q_Mars():
    def __init__(self,config):
        print(f"Using device: {device}")
        
        self.config = config
        self.game_window = GameWindow(800)
        self.size = self.config["size"]

        input_size = 11
        hidden_size = 128
        output_size = len(Actions)

        self.network = DQN(input_size, hidden_size, output_size).to(device)
        self.target_network = DQN(input_size, hidden_size, output_size).to(device)

        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01)

        self.robot = Robot()
        self.rocks = self.config["rocks"]
        self.uphills = self.config["uphills"]
        self.downhills = self.config["downhills"]
        self.transmiter_stations = self.config["transmiter_stations"]
        self.batery_stations = self.config["batery_stations"]

    def run(self):
        self.game_window.render_game(self)
    
    def reset(self):
        self.robot.position = (0, 0)
        self.robot.battery = 100
        self.robot.holding_rock_count = 0
        self.rocks = self.config["rocks"]

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

        if self.robot.battery < 50 and self.robot.position in self.batery_stations:
            possible_actions.append(Actions.RECHARGE)

        if self.robot.position in self.rocks:
            possible_actions.append(Actions.COLLECT)

        print(f"Possible actions: {possible_actions}")

        action = self.network.forward()

        return action
    
    def step(self, action):
        old_position = self.robot.position
        self.update_robot(action)
        next_state = self.robot.position
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

        if self.robot.current_position in self.rocks and self.robot.current_position == old_position:
            reward += 150
        elif self.robot.current_position in self.uphills:
            reward += -10
        elif self.robot.current_position in self.downhills:
            reward += 1
        elif self.robot.current_position in self.transmiter_stations and self.robot.current_position == old_position:
            reward += 250
        elif self.robot.current_position in self.batery_stations and self.robot.current_position == old_position:
            reward += 100
        elif self.robot.current_position in self.cliffs:
            reward += -200

        return reward
    

    def calculate_surrounding(self):
        position = self.robot.position

        obs = np.zeros( (3, 3), dtype = np.int8)

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

        for batery_station in self.batery_stations:
            if (position[0] - 1, position[1] - 1) == batery_station:
                obs[0, 0] = Entities.BATTERY_STATION.value
            if (position[0] - 1, position[1]) == batery_station:
                obs[0, 1] = Entities.BATTERY_STATION.value
            if (position[0] - 1, position[1] + 1) == batery_station:
                obs[0, 2] = Entities.BATTERY_STATION.value
            if (position[0], position[1] - 1) == batery_station:
                obs[1, 0] = Entities.BATTERY_STATION.value
            if (position[0], position[1] + 1) == batery_station:
                obs[1, 2] = Entities.BATTERY_STATION.value
            if (position[0] + 1, position[1] - 1) == batery_station:
                obs[2, 0] = Entities.BATTERY_STATION.value
            if (position[0] + 1, position[1]) == batery_station:
                obs[2, 1] = Entities.BATTERY_STATION.value
            if (position[0] + 1, position[1] + 1) == batery_station:
                obs[2, 2] = Entities.BATTERY_STATION.value             
                
        obs[1, 1] = Entities.ROBOT.value
                
        return obs