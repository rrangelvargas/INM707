import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from utils import Actions, Entities, Robot, Transition, device
from game_window import GameWindow
import matplotlib.pyplot as plt
import os
import json

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = "dqn_policy_net.pth"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

def state_to_tensor(state_vector):
    return torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)

class MarsEnv: # the mars environment class
    def __init__(self, config):
        self.size = config["size"]
        self.rocks = config.get("rocks", []).copy()
        self.transmitter_stations = config.get("transmitter_stations", []).copy()
        self.cliffs = config.get("cliffs", []).copy()
        self.uphills = config.get("uphills", []).copy()
        self.downhills = config.get("downhills", []).copy()
        self.battery_stations = config.get("battery_stations", []).copy()
        self.termination_reason = None
        self.reset()

    def reset(self): # resets the robot after each episode
        self.robot_position = [0, 0]
        self.robot_battery = 100
        self.robot_holding = 0
        self.current_rocks = self.rocks.copy()
        return self.get_state()

    def get_state(self): # returns the current state of the environment
        grid = np.full((self.size, self.size), Entities.EMPTY.value, dtype=np.int8)
        for x, y in self.current_rocks:
            grid[y, x] = Entities.ROCK.value
        for x, y in self.transmitter_stations:
            grid[y, x] = Entities.TRANSMITTER_STATION.value
        for x, y in self.cliffs:
            grid[y, x] = Entities.CLIFF.value
        for x, y in self.uphills:
            grid[y, x] = Entities.UPHILL.value
        for x, y in self.downhills:
            grid[y, x] = Entities.DOWNHILL.value
        for x, y in self.battery_stations:
            grid[y, x] = Entities.BATTERY_STATION.value
        rx, ry = self.robot_position
        grid[ry, rx] = Entities.ROBOT.value
        flat = grid.flatten().astype(np.float32) / float(len(Entities)) # flatten and normalize the grid to be between 0 and 1
        
        min_rock_distance = float('inf')
        for rock_x, rock_y in self.current_rocks: # loops over the rocks
            distance = abs(rx - rock_x) + abs(ry - rock_y)  # finds the distance between robot and rock
            min_rock_distance = min(min_rock_distance, distance)  # updates min_rock_distance if distance is smaller than last distance
        if not self.current_rocks: # if there are no rocks
            min_rock_distance = 0  # set min_rock_distance to 0
        rock_distance = np.array([min_rock_distance / (self.size * 2)], dtype=np.float32)  # normalize by max possible distance to a rock
        tx, ty = self.transmitter_stations[0]  # assuming there is only one transmitter
        transmitter_distance = abs(rx - tx) + abs(ry - ty)
        transmitter_distance = np.array([transmitter_distance / (self.size * 2)], dtype=np.float32)  # Normalize 
        rocks_remaining = np.array([len(self.current_rocks) / max(1, len(self.rocks))], dtype=np.float32) # number of rocks remaining is normalized
        battery_level = np.array([self.robot_battery / 100.0], dtype=np.float32) # the battery level is normalized
        return np.concatenate([flat, rock_distance, transmitter_distance, rocks_remaining, battery_level]) # the state vector is concatenated based on these values

    def get_possible_actions(self): # returns a list of possible actions for the current state
        acts = []
        x, y = self.robot_position
        if x > 0: acts.append(Actions.LEFT) # if the robot can move left
        if x < self.size - 1: acts.append(Actions.RIGHT) # if the robot can move right
        if y > 0: acts.append(Actions.UP) # if the robot can move up
        if y < self.size - 1: acts.append(Actions.DOWN) # if the robot can move down
        if self.robot_position in self.current_rocks: acts.append(Actions.COLLECT) # if the robot can collect a rock
        if self.robot_position in self.battery_stations and self.robot_battery < 100: acts.append(Actions.RECHARGE) # if the robot can recharge
        if self.robot_holding == 3 and self.robot_position in self.transmitter_stations: acts.append(Actions.TRANSMIT) # if the robot can transmit
        return acts if acts else [Actions.RIGHT] # if there are no possible actions, return [Actions.RIGHT]

    def step(self, action):
        reward = -1  # the penalty for each step moved

        self.robot_battery -= 1  # decreases the battery by 1
        self.termination_reason = None  # resets the termination reason
        if action == Actions.RIGHT: # if the action is to move right
            self.robot_position[0] += 1
        elif action == Actions.LEFT: # if the action is to move left
            self.robot_position[0] -= 1
        elif action == Actions.UP: # if the action is to move up
            self.robot_position[1] -= 1
        elif action == Actions.DOWN: # if the action is to move down
            self.robot_position[1] += 1
        elif action == Actions.COLLECT: # if the action is to collect a rock
            self.robot_holding += 1 # increment the number of rocks held by one
            self.current_rocks.remove(self.robot_position) # remove the rock from the list of rocks
            if self.robot_holding == 1:
                reward += 100 # 100 reward for collecting a rock
            elif self.robot_holding == 2:
                reward += 200 # 200 reward for collecting two rocks
            elif self.robot_holding == 3:
                reward += 300 # 300 reward for collecting three rocks
        elif action == Actions.RECHARGE: # if the action is to recharge
            self.robot_battery = 100  # set the battery back to 100
            reward += 50  # reward for recharging 
        elif action == Actions.TRANSMIT:
            reward += 500  # reward for transmitting
        if self.robot_position in self.cliffs:
            reward -= 100  # penalty for falling into a cliff
        if self.robot_battery <= 0:
            reward -= 100  # penalty for running out of battery

        # step_penalty = -0.1 * (1 + self.robot_holding * 0.2)
        # reward += step_penalty

        done_battery = (self.robot_battery <= 0) # if the battery is zero done battery is true
        done_cliff   = (self.robot_position in self.cliffs) # if the robot is in a cliff done cliff is true
        done_goal    = (action == Actions.TRANSMIT) # if the action is to transmit done goal is true
        done         = done_battery or done_cliff or done_goal # if done battery or done cliff or done goal is true done is true

        if done: # if the robot is done
            if done_battery: # check if the robot ran out of battery
                self.termination_reason = "battery" # set the termination reason to battery
            elif done_cliff: # check if the robot fell into a cliff
                self.termination_reason = "cliff" # set the termination reason to cliff
            else: # if the robot did not run out of battery or fall into a cliff
                self.termination_reason = "goal_reached" # set the termination reason to goal reached
        else: # if the robot is not done
            self.termination_reason = None # set the termination reason to None

        next_state = None if done else self.get_state() # if the robot is not call the get state function
        return next_state, reward, done

class DQN(nn.Module): # the deep q network class
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # the forward pass function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory: # the simple replay memory class
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition): # adds the new experience to the buffer
        if len(self.memory) < self.capacity: # if the buffer is full
            self.memory.append(None) # add a None to the buffer
        self.memory[self.position] = transition # add the new experience to the buffer
        self.position = (self.position + 1) % self.capacity # increment the position

    def sample(self, batch_size): # samples a batch of experiences from the buffer
        return random.sample(self.memory, batch_size)

    def __len__(self): # returns the length of the buffer
        return len(self.memory)

class PrioritizedReplayMemory: # the prioritized replay memory class
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.position = 0

    def push(self, transition, priority=None): # adds the new experience to the buffer
        max_priority = max(self.priorities, default=1.0)
        if priority is None:
            priority = max_priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4): # samples a batch of experiences from the buffer
        priorities = np.array(self.priorities) # converts the priorities to probabilities
        probs = priorities ** self.alpha # raises the priorities to the power of alpha
        probs /= probs.sum() # normalizes the probabilities

        indices = np.random.choice(len(self.buffer), batch_size, p=probs) # sampling the transitions based on the probabilities
        samples = [self.buffer[i] for i in indices] # getting the sampled transitions

        total = len(self.buffer) # finds the total number of transitions
        weights = (total * probs[indices]) ** (-beta) # calcualting the importance sampling weights
        weights /= weights.max()  # normalizing the weights

        return samples, indices, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, indices, priorities): # updates the priorities based on the td error of the transitions
        for idx, priority in zip(indices, priorities): # looping over the indices and priorities
            self.priorities[idx] = priority.item()

    def __len__(self): # returns the length of the buffer
        return len(self.buffer)

class DeepQAgent:
    def __init__(self, env,
                 display=False,
                 params=None):
        
        self.env = env
        self.display = display
        self.policy_type = params["policy_type"]

        self.loss_history = []
        self.update_count = 0
        self.success_history = []

        self.block_success_lengths = []
        self.block_success_count = 0

        self.success_count = 0
        self.timeout_count = 0
        self.cliff_fall_count = 0
        self.battery_empty_count = 0

        self.episode_rewards = []
        self.episode_lengths  = []
        self.steps_to_success = []

        self.gamma         = float(params["gamma"])
        self.batch_size    = int(params["batch_size"])
        self.epsilon       = float(params["epsilon_start"])
        self.epsilon_decay = float(params["epsilon_decay"])
        self.epsilon_min   = float(params["epsilon_min"])
        self.learning_rate = float(params["learning_rate"])

        self.memory = ReplayMemory(params["memory_capacity"]) if not params["prioritised"] else PrioritizedReplayMemory(params["memory_capacity"], params["per_alpha"])

        input_size = self.env.size * self.env.size + 4  # grid (size*size) + rock_distance (1) + transmitter_distance (1) + rocks_remaining (1) + battery_level (1)
        hidden_size = params["hidden_size"] # hidden size is set based on the params
        output_size = len(Actions) # output size is set to the number of actions

        self.policy_net = DQN(input_size, hidden_size, output_size).to(device) # defines the policy network
        self.target_net = DQN(input_size, hidden_size, output_size).to(device) # defines the target network
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copies the weights of the policy network to the target network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate) # defines the optimizer

        if self.display:
            pygame.init()
            self.game_window = GameWindow(800)

    def select_action(self, state_vector, possible_actions): # selects an action
        state_tensor = state_to_tensor(state_vector)

        if self.policy_type == 'epsilon_greedy': # if the policy type is epsilon_greedy
            if random.random() < self.epsilon: # if a random number is less than epsilon
                return random.choice(possible_actions) # returns a random action
            with torch.no_grad(): # disables gradient calculation
                q_values = self.policy_net(state_tensor)
                action_idxs = torch.tensor(
                    [a.value for a in possible_actions],
                    device=q_values.device,
                    dtype=torch.long
                )
                allowed_q = q_values[0].gather(0, action_idxs)
                best_idx = torch.argmax(allowed_q).item()
        
            return possible_actions[best_idx]
        elif self.policy_type == 'softmax': # if the policy type is softmax
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idxs = torch.tensor(
                    [a.value for a in possible_actions],
                    device=q_values.device,
                    dtype=torch.long
                )
                allowed_q = q_values[0].gather(0, action_idxs)
                # Softmax over allowed_q
                probs = torch.softmax(allowed_q, dim=0).cpu().numpy()
                chosen_idx = np.random.choice(len(possible_actions), p=probs)
            return possible_actions[chosen_idx]
        else:
            raise ValueError(f"Unknown policy_type: {self.policy_type}")

    def optimize_model(self, double_dqn, prioritised, beta): # optimizes the model
        if len(self.memory) < self.batch_size: # if the buffer is not full
            return

        if prioritised: # if prioritised is set
            transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        else: # if not using prioritised
            transitions = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size, dtype=torch.float32, device=device)  # default to 1s if not prioritised

        batch = Transition(*zip(*transitions)) # unpacks the transitions

        state_batch = torch.cat(batch.state) # concatenates the states
        action_batch = torch.cat(batch.action) # concatenates the actions
        reward_batch = torch.cat(batch.reward) # concatenates the rewards

        non_final_mask = torch.tensor( # creates a mask for the non-final states
            tuple(s is not None for s in batch.next_state),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat( # concatenates the non-final states
            [s for s in batch.next_state if s is not None]
        )

        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # gets the state-action values based on the policy net
        next_state_values = torch.zeros(self.batch_size, device=device) # creates a tensor for the next state values

        if not double_dqn: # if not using double dqn
            with torch.no_grad():
                next_q = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = next_q.max(1)[0]
        else: # if using double dqn
            if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    policy_q_next = self.policy_net(non_final_next_states)  # selects the best action from policy net
                    best_action_idxs = policy_q_next.argmax(dim=1, keepdim=True)  # gets the indices of the best actions
                    target_q_next = self.target_net(non_final_next_states)  # selects the best action from target net
                    selected_q = target_q_next.gather(1, best_action_idxs).squeeze(1)  # gets the selected q values
                next_state_values[non_final_mask] = selected_q  # sets the next state values

        expected_state_action_values = reward_batch + self.gamma * next_state_values # gets the expected state-action values

        td_errors = (expected_state_action_values - state_action_values.squeeze()).detach() # finds the td errors
        losses = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values, reduction='none') # gets the losses using the l1 loss

        loss = (losses * weights).mean() # calulates the loss from the losses and the weights

        self.loss_history.append(loss.item()) # adds the loss to the loss history
        self.optimizer.zero_grad() # zeros out the gradients
        loss.backward() # computes the gradients
        self.optimizer.step() # updates the weights

        if prioritised: # if using per 
            new_priorities = td_errors.abs() + 1e-5
            self.memory.update_priorities(indices, new_priorities) # updates the priorities

    def update_epsilon(self): # updating epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self, tau): # updates the target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def _render(self, episode: int, step: int): # renders the environment
        grid_sz = int(self.game_window.window_size / self.env.size)
        pygame.draw.rect(
            self.game_window.display,
            self.game_window.GRID_COLOR,
            (self.game_window.sidebar_width, 0,
             self.game_window.window_size, self.game_window.window_size)
        )

        self.game_window.draw_grid(grid_sz) # draws the grid
        self.game_window.draw_sidebar( # draws the sidebar
            episode,
            step,
            self.env.robot_battery,
            epsilon=self.epsilon,
            policy="epsilon_greedy"
        )

        vis = Robot() # creates a robot
        vis.position = self.env.robot_position.copy() # sets the robot's position
        vis.battery  = self.env.robot_battery  # sets the robot's battery
        vis.holding_rock_count = self.env.robot_holding  # sets the robot's holding rock count
        self.game_window.render_images( # rendering the images
            vis,
            self.env.current_rocks,
            self.env.transmitter_stations,
            self.env.cliffs,
            self.env.uphills,
            self.env.downhills,
            self.env.battery_stations,
            grid_sz
        )
        pygame.display.flip()

    def train(self, params, run_name, save_graphs=False): # the main training loop
        num_episodes     = params["num_episodes"] # number of episodes
        max_steps        = params["max_steps"]    # maximum number of steps per episode
        target_update    = params["target_update"] # number of steps between target network updates
        block_length     = params["block_length"] # number of steps before a block ends

        print("Using Double DQN" if params["double_dqn"] else "Using Standard DQN") # prints whether or not double dqn is being used
        print("Using Prioritised Experience Replay" if {params['prioritised']} else "Not Using Prioritised Experience Replay") # prints whether or not prioritised experience replay is being used

        for ep in range(1, num_episodes + 1): # for loop for episode length
            state = self.env.reset()
            total_reward, done = 0, False

            for step in range(1, max_steps + 1): # for loop for step length
                action = self.select_action(state, self.env.get_possible_actions()) # selects an action
                next_state, reward, done = self.env.step(action) # steps the environment
                total_reward += reward # adds the reward to the total reward

                self.memory.push(Transition( # adds the transition to the memory
                    state_to_tensor(state),
                    torch.tensor([[action.value]], device=device),
                    None if next_state is None else state_to_tensor(next_state),
                    torch.tensor([reward], device=device)
                ))
                self.optimize_model(params["double_dqn"], params["prioritised"], params["per_beta"]) # optimizes the model based on the settings set

                state = next_state # sets the state to the next state

                if self.display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    self._render(ep, step)

                if step == max_steps: # if the max steps have been reached
                    self.env.termination_reason = "timeout"
                    done = True

                if done: # if the episode is done
                    reason = self.env.termination_reason # gets the termination reason
                    if reason == "battery":
                        self.battery_empty_count += 1
                    elif reason == "cliff":
                        self.cliff_fall_count += 1
                    elif reason == "goal_reached":
                        self.success_count += 1
                        self.block_success_count += 1
                        self.steps_to_success.append(step)
                        self.block_success_lengths.append(step)
                    elif reason == "timeout":
                        self.timeout_count += 1
                    break

            self.update_epsilon() # updates the epsilon
            if ep % target_update == 0: # if the target update interval has been reached
                self.update_target(params["tau"]) # updates the target with tau

            self.episode_rewards.append(total_reward) # adds the total reward to the episode rewards
            self.episode_lengths.append(step) # adds the step to the episode lengths

            if ep % block_length == 0: # prints block summary
                block_rewards = self.episode_rewards[-block_length:]
                block_lengths = self.episode_lengths[-block_length:]
                block_avg_r = sum(block_rewards) / len(block_rewards)
                block_avg_len = sum(block_lengths) / len(block_lengths)
                avg_succ_len = sum(self.block_success_lengths) / len(self.block_success_lengths) if len(self.block_success_lengths) > 0 else 0
                self.success_history.append(self.block_success_count)
                last_loss = self.loss_history[-1] if self.loss_history else float('nan')
                print(f"\n=== Episodes {ep-(block_length-1):4d} {ep-1:4d} summary ===")
                print(f"Avg Reward: {block_avg_r:.2f} | "f"Loss {last_loss:.4f} | "f"Avg Episode Length: {block_avg_len:.2f} | "f"Avg Successful Episode Length: {avg_succ_len:.2f}\n")
                
                # Check for early stopping
                current_success_rate = (self.block_success_count / block_length) * 100
                # if current_success_rate >= 90:
                #     print(f"\nEarly stopping triggered! Success rate of {current_success_rate:.2f}% exceeds 90% threshold.")
                #     break
                
                self.block_success_lengths = []
                self.block_success_count = 0

            # prints final results
            print(f"Episode {ep:4d}: Reward={total_reward:.2f}, Length={step}, Epsilon={self.epsilon:.4f}, Termination Reason={self.env.termination_reason}, Rocks Collected={self.env.robot_holding}")

        num_blocks = len(self.episode_lengths) // block_length # number of blocks
        block_indices = list(range(1, len(self.success_history) + 1)) # list of block indices
        block_success_percentages = [ (cnt / block_length) * 100 for cnt in self.success_history ] # list of block success percentages
        block_avg_lengths = [ sum(self.episode_lengths[i*block_length:(i+1)*block_length]) / block_length for i in range(num_blocks)]  # list of block average lengths  
        
        print(f"Overall Avg Reward: {sum(self.episode_rewards) / len(self.episode_rewards):.2f}")
        print(f"Overall Avg Steps: {sum(self.episode_lengths) / len(self.episode_lengths):.2f}")
        print(f"Overall Avg Steps to Success: {sum(self.steps_to_success) / len(self.steps_to_success) if len(self.steps_to_success) > 0 else 0}")
        print("Overall Cliff Rate: ", self.cliff_fall_count / num_episodes)
        print("Overall Success Rate: ", self.success_count / num_episodes)
        print("Overall Failed Rate: ", self.timeout_count / num_episodes)
        print("Overall Battery Empty Rate: ", self.battery_empty_count / num_episodes)


        if not os.path.exists(f'output/{run_name}'):
            os.makedirs(f'output/{run_name}')

        if save_graphs: # if the graphs are to be saved
            plt.figure()
            plt.plot(self.loss_history)
            plt.xlabel('Optimization step')
            plt.ylabel('TD-error loss')
            plt.title('DQN Training Loss Curve')
            plt.savefig(f'output/{run_name}/dqn_loss_curve.png')
            plt.close()

            plt.figure()
            plt.bar(block_indices, block_success_percentages)
            plt.xlabel("Block Number")
            plt.ylabel("Success Rate (%)")
            plt.title("Block Success Rate (%) Over Blocks")
            plt.xticks(block_indices)
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.savefig(f'output/{run_name}/block_success_rate.png')
            plt.close()

            plt.figure()
            plt.bar(block_indices, block_avg_lengths)
            plt.xlabel("Block Number")
            plt.ylabel("Average Episode Length")
            plt.title("Average Episode Length per Block")
            plt.xticks(block_indices)
            plt.tight_layout()
            plt.savefig(f'output/{run_name}/block_avg_episode_length.png')
            plt.close()

    def save_results(self, run_name, num_episodes, save_checkpoint): # saves the results to a json file
        results = {
            "overall_avg_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "overall_avg_steps": sum(self.episode_lengths) / len(self.episode_lengths),
            "overall_avg_steps_to_success": sum(self.steps_to_success) / len(self.steps_to_success) if len(self.steps_to_success) > 0 else 0,
            "overall_cliff_rate": self.cliff_fall_count / num_episodes,
            "overall_success_rate": self.success_count / num_episodes,
            "overall_failed_rate": self.timeout_count / num_episodes,
            "overall_battery_empty_rate": self.battery_empty_count / num_episodes
        }

        with open(f'output/{run_name}/results_{num_episodes}.json', "w") as f:    
            json.dump(results, f)

        if save_checkpoint:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_{num_episodes}.pth")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(self.policy_net.state_dict(), checkpoint_path)
            print(f"Saved policy network weights to {checkpoint_path}")
