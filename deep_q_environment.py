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

HPARAMS = {
    "num_episodes":    10000,          
    "max_steps":       500,
    "target_update":   1000,         
    "memory_capacity": 5000,      
    "batch_size":      128,          # Reduced batch size for faster learning
    "hidden_size":     256,          
    "learning_rate":   1e-4,         # Increased learning rate
    "gamma":           0.99,         # Increased gamma for better long-term planning
    "epsilon_start":   1.0,
    "epsilon_decay":   0.995,        # Faster epsilon decay
    "epsilon_min":     0.01,
    "tau":             0.005,        # Soft update parameter
}

# Convert flat state vector to tensor
def state_to_tensor(state_vector):
    return torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)

class MarsEnv:
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

    def reset(self):
        self.robot_position = [0, 0]
        self.robot_battery = 100
        self.robot_holding = 0
        self.current_rocks = self.rocks.copy()
        return self.get_state()

    def get_state(self):
        grid = np.full((self.size, self.size), Entities.EMPTY.value, dtype=np.int8)
        for x, y in self.current_rocks:
            grid[y, x] = Entities.ROCK.value
        for x, y in self.transmitter_stations:
            grid[y, x] = Entities.TRANSMITER_STATION.value
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
        flat = grid.flatten().astype(np.float32) / float(len(Entities))
        
        # Calculate distance to nearest rock
        min_rock_distance = float('inf')
        for rock_x, rock_y in self.current_rocks:
            distance = abs(rx - rock_x) + abs(ry - rock_y)  # Manhattan distance
            min_rock_distance = min(min_rock_distance, distance)
        if not self.current_rocks:  # If no rocks left
            min_rock_distance = 0
        rock_distance = np.array([min_rock_distance / (self.size * 2)], dtype=np.float32)  # Normalize by max possible distance
        
        # Number of rocks remaining normalized by total initial rocks
        rocks_remaining = np.array([len(self.current_rocks) / max(1, len(self.rocks))], dtype=np.float32)
        
        return np.concatenate([flat, rock_distance, rocks_remaining])

    def get_possible_actions(self):
        acts = []
        x, y = self.robot_position
        if x > 0: acts.append(Actions.LEFT)
        if x < self.size - 1: acts.append(Actions.RIGHT)
        if y > 0: acts.append(Actions.UP)
        if y < self.size - 1: acts.append(Actions.DOWN)
        if self.robot_position in self.current_rocks: acts.append(Actions.COLLECT)
        # if self.robot_position in self.battery_stations and self.robot_battery < 100: acts.append(Actions.RECHARGE)
        if self.robot_holding == 3 and self.robot_position in self.transmitter_stations: acts.append(Actions.TRANSMIT)
        return acts if acts else [Actions.RIGHT]

    def step(self, action):
        reward = -0.1  # Small step penalty to encourage efficient paths
        self.termination_reason = None
        if action == Actions.RIGHT:
            self.robot_position[0] += 1
        elif action == Actions.LEFT:
            self.robot_position[0] -= 1
        elif action == Actions.UP:
            self.robot_position[1] -= 1
        elif action == Actions.DOWN:
            self.robot_position[1] += 1
        elif action == Actions.COLLECT:
            self.robot_holding += 1
            self.current_rocks.remove(self.robot_position)
            if self.robot_holding == 1:
                reward += 100  # Keep rock collection rewards high
            elif self.robot_holding == 2:
                reward += 200
            elif self.robot_holding == 3:
                reward += 300
        elif action == Actions.RECHARGE:
            self.robot_battery = 100
        elif action == Actions.TRANSMIT:
            reward += 500  # Keep transmission reward high
        if self.robot_position in self.cliffs:
            reward -= 100  # Keep cliff penalty high

        # Add a small penalty for each step taken, scaled by the number of rocks collected
        # This encourages the agent to find efficient paths while still prioritizing rock collection
        step_penalty = -0.1 * (1 + self.robot_holding * 0.2)  # Penalty increases slightly with each rock collected
        reward += step_penalty

        done_battery = (self.robot_battery <= 0)
        done_cliff   = (self.robot_position in self.cliffs)
        done_goal    = (action == Actions.TRANSMIT)
        done         = done_battery or done_cliff or done_goal

        if done:
            if done_battery:
                self.termination_reason = "battery"
            elif done_cliff:
                self.termination_reason = "cliff"
            else:  # must be goal_reached
                self.termination_reason = "goal_reached"
        else:
            self.termination_reason = None

        next_state = None if done else self.get_state()
        return next_state, reward, done

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DeepQAgent:
    def __init__(self, env, learning_rate=1e-4,
                 memory_capacity=10000,
                 display=False):
        
        self.env = env
        self.display = display
        self.loss_history = []

        p = HPARAMS
        self.gamma        = p["gamma"]
        self.batch_size   = p["batch_size"]
        self.epsilon      = p["epsilon_start"]
        self.epsilon_decay= p["epsilon_decay"]
        self.epsilon_min  = p["epsilon_min"]
        self.memory       = ReplayMemory(p["memory_capacity"])

        self.memory = ReplayMemory(memory_capacity)

        input_size = env.size * env.size + 2
        hidden_size = p["hidden_size"]
        output_size = len(Actions)

        self.policy_net = DQN(input_size, hidden_size, output_size).to(device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if self.display:
            pygame.init()
            self.game_window = GameWindow(800)

    def select_action(self, state_vector, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        with torch.no_grad():
            q_values = self.policy_net(state_to_tensor(state_vector)).cpu().numpy().flatten()
            return max(possible_actions, key=lambda a: q_values[a.value])

    def optimize_model(self):
        # Only learn once we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create tensors for states, actions, rewards, and next_states
        state_batch      = torch.cat(batch.state)
        action_batch     = torch.cat(batch.action)
        reward_batch     = torch.cat(batch.reward)
        non_final_mask   = torch.tensor(
                            tuple(s is not None for s in batch.next_state),
                            device=device,
                            dtype=torch.bool
                        )
        non_final_next_states = torch.cat(
                            [s for s in batch.next_state if s is not None]
                        )

        # Compute current Q values: Q(s_t, a_t) with policy network
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for non-final next states using target network
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_q = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = next_q.max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        # Compute Huber (smooth L1) loss
        loss = F.smooth_l1_loss(
            state_action_values.squeeze(), 
            expected_state_action_values
        )

        # Record the loss for plotting later
        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # (Optional) clip gradients here if you wish:
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        # Soft update of target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                HPARAMS["tau"] * policy_param.data + (1.0 - HPARAMS["tau"]) * target_param.data
            )

    def _render(self, episode: int, step: int):
        grid_sz = int(self.game_window.window_size / self.env.size)
        pygame.draw.rect(
            self.game_window.display,
            self.game_window.GRID_COLOR,
            (self.game_window.sidebar_width, 0,
             self.game_window.window_size, self.game_window.window_size)
        )

        self.game_window.draw_grid(grid_sz)
        self.game_window.draw_sidebar(
            episode,
            step,
            self.env.robot_battery,
            epsilon=self.epsilon,
            policy="epsilon_greedy"
        )
        # render robot via utils.Robot
        vis = Robot()
        vis.position = self.env.robot_position.copy()
        vis.battery  = self.env.robot_battery
        vis.holding_rock_count = self.env.robot_holding
        self.game_window.render_images(
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

    def train(self):
        p = HPARAMS
        num_episodes     = p["num_episodes"]
        max_steps        = p["max_steps"]
        target_update    = p["target_update"]
        episode_rewards = []
        episode_lengths  = []
        cliff_fall_count = 0
        success_count    = 0
        battery_empty_count = 0
        steps_to_success = []
        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            total_reward, done = 0, False

            for t in range(1, max_steps + 1):
                action = self.select_action(state, self.env.get_possible_actions())
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # push & optimize
                self.memory.push(Transition(
                    state_to_tensor(state),
                    torch.tensor([[action.value]], device=device),
                    None if next_state is None else state_to_tensor(next_state),
                    torch.tensor([reward], device=device)
                ))
                self.optimize_model()

                state = next_state

                if done:
                    if self.env.termination_reason == "cliff":
                        cliff_fall_count += 1
                    elif self.env.termination_reason == "goal_reached":
                        success_count += 1
                        steps_to_success.append(t)
                    else:
                        battery_empty_count += 1
                    break
                elif t == max_steps:
                    self.env.termination_reason = "timeout"
                    break

            # per-episode updates
            self.update_epsilon()
            if ep % target_update == 0:
                self.update_target()

            # record metrics for this episode
            episode_rewards.append(total_reward)
            episode_lengths.append(t)

            # every 100 episodes, print a summary
            if ep % 100 == 0:
                last100_rewards = episode_rewards[-100:]
                last100_lengths = episode_lengths[-100:]
                avg_r = sum(last100_rewards) / len(last100_rewards)
                avg_len = sum(last100_lengths) / len(last100_lengths)
                print(f"=== Episodes {ep-99:4d} {ep:4d} summary ===")
                print(f"Avg Reward: {avg_r:.2f} | Loss {self.loss_history[-1]:.4f}")
            print(f"Episode {ep:4d}: Reward={total_reward:.2f}, Length={t}, Epsilon={self.epsilon:.4f}, Termination Reason={self.env.termination_reason}, Rocks Collected={self.env.robot_holding}")

        print("Cliff Fall Rate: ", cliff_fall_count / num_episodes)
        print("Success Rate: ", success_count / num_episodes)
        print("Failed Rate: ", (num_episodes - success_count - cliff_fall_count) / num_episodes)
        print("Battery Empty Rate: ", battery_empty_count / num_episodes)


        plt.figure()
        plt.plot(agent.loss_history)
        plt.xlabel('Optimization step')
        plt.ylabel('TD-error loss')
        plt.title('DQN Training Loss Curve')
        plt.show()

if __name__ == "__main__":
    config = {"size":5, "rocks":[[1,2],[3,3],[2,4]],
              "transmitter_stations":[[4,4]], "cliffs":[[2,3],[1,1]],
              "uphills":[[0,4],[2,0]], "downhills":[[3,0],[0,2]],
              "battery_stations":[[4,2]]}
    env = MarsEnv(config)
    agent = DeepQAgent(env, display=False)
    agent.train()