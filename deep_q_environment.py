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

params = {
    "num_episodes":    200,
    "block_length":    20,          
    "max_steps":       500,
    "target_update":   1000,         
    "memory_capacity": 5000,      
    "batch_size":      128,
    "hidden_size":     256,          
    "learning_rate":   1e-4,
    "gamma":           0.99,
    "epsilon_start":   1.0,
    "epsilon_decay":   0.995,
    "epsilon_min":     0.01,
    "tau":             0.005,
    "display":         False,
    "double_dqn":      False
}

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
        flat = grid.flatten().astype(np.float32) / float(len(Entities))
        
        min_rock_distance = float('inf')
        for rock_x, rock_y in self.current_rocks:
            distance = abs(rx - rock_x) + abs(ry - rock_y)
            min_rock_distance = min(min_rock_distance, distance)
        if not self.current_rocks:
            min_rock_distance = 0
        rock_distance = np.array([min_rock_distance / (self.size * 2)], dtype=np.float32)
        rock_distance = np.array([min_rock_distance / (self.size * 2)], dtype=np.float32)  # Normalize by max possible distance
        
        # Calculate distance to transmitter
        tx, ty = self.transmitter_stations[0]  # Assuming one transmitter
        transmitter_distance = abs(rx - tx) + abs(ry - ty)
        transmitter_distance = np.array([transmitter_distance / (self.size * 2)], dtype=np.float32)  # Normalize
        
        # Number of rocks remaining normalized by total initial rocks
        rocks_remaining = np.array([len(self.current_rocks) / max(1, len(self.rocks))], dtype=np.float32)
        return np.concatenate([flat, rock_distance, rocks_remaining])
        
        return np.concatenate([flat, rock_distance, transmitter_distance, rocks_remaining])

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
        reward = -0.1
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
                reward += 100
            elif self.robot_holding == 2:
                reward += 200
            elif self.robot_holding == 3:
                reward += 300
        elif action == Actions.RECHARGE:
            self.robot_battery = 100
        elif action == Actions.TRANSMIT:
            reward += 500
        if self.robot_position in self.cliffs:
            reward -= 100

        step_penalty = -0.1 * (1 + self.robot_holding * 0.2)
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
            else:
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
    def __init__(self, env,
                 display=False):
        
        self.env = env
        self.display = display

        self.loss_history = []
        self.update_count = 0  # Add counter for update tracking
        self.success_history = []

        self.block_success_lengths = []
        self.block_success_count = 0

        self.success_count = 0
        self.timeout_count = 0
        self.cliff_fall_count = 0

        self.episode_rewards = []
        self.episode_lengths  = []
        self.steps_to_success = []

        self.gamma        = params["gamma"]
        self.batch_size   = params["batch_size"]
        self.epsilon      = params["epsilon_start"]
        self.epsilon_decay= params["epsilon_decay"]
        self.epsilon_min  = params["epsilon_min"]
        self.learning_rate= params["learning_rate"]

        self.memory = ReplayMemory(params["memory_capacity"])

        # Update input size to account for transmitter_distance
        input_size = env.size * env.size + 3  # +3 for rock_distance, transmitter_distance, and rocks_remaining
        hidden_size = params["hidden_size"]
        output_size = len(Actions)

        self.policy_net = DQN(input_size, hidden_size, output_size).to(device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        if self.display:
            pygame.init()
            self.game_window = GameWindow(800)

    def select_action(self, state_vector, possible_actions):
        state_tensor = state_to_tensor(state_vector)

        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_idxs = torch.tensor(
                [a.value for a in possible_actions],
                device=q_values.device,
                dtype=torch.long
            )
            allowed_q = q_values[0].gather(0, action_idxs)
            best_idx = torch.argmax(allowed_q).item()
    
        return possible_actions[best_idx]

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        double_dqn = params["double_dqn"]
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)

        if not double_dqn: # if not using double dqn
            with torch.no_grad():
                next_q = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = next_q.max(1)[0]
        else: # if using double dqn
           if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    # 1) using the policy net to pick best actions in next states
                    policy_q_next = self.policy_net(non_final_next_states)
                    best_action_idxs = policy_q_next.argmax(dim=1, keepdim=True)
                    # 2) evaluate those actions with target_net
                    target_q_next = self.target_net(non_final_next_states)
                    selected_q = target_q_next.gather(1, best_action_idxs).squeeze(1)
                next_state_values[non_final_mask] = selected_q 
        
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values.squeeze(), 
            expected_state_action_values
        )

        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                params["tau"] * policy_param.data + (1.0 - params["tau"]) * target_param.data
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
        num_episodes     = params["num_episodes"]
        max_steps        = params["max_steps"]
        target_update    = params["target_update"]
        block_length     = params["block_length"]
        CHECKPOINT_DIR = "checkpoints"
        CHECKPOINT_FILE = "dqn_policy_net.pth"
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

        print("Using Double DQN" if params["double_dqn"] else "Using Standard DQN")

        if os.path.isdir(CHECKPOINT_DIR) and os.path.isfile(CHECKPOINT_PATH):
            self.policy_net.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=device))
            print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        else:
            print(f"No checkpoint found at {CHECKPOINT_PATH}, starting fresh")


        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            total_reward, done = 0, False

            for step in range(1, max_steps + 1):
                action = self.select_action(state, self.env.get_possible_actions())
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                self.memory.push(Transition(
                    state_to_tensor(state),
                    torch.tensor([[action.value]], device=device),
                    None if next_state is None else state_to_tensor(next_state),
                    torch.tensor([reward], device=device)
                ))
                self.optimize_model()

                state = next_state

                if self.display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    self._render(ep, step)

                if step == max_steps:
                    self.env.termination_reason = "timeout"
                    done = True

                if done:
                    reason = self.env.termination_reason
                    if   reason == "battery":
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

            self.update_epsilon()
            if ep % target_update == 0:
                self.update_target()

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)

            if ep % block_length == 0:
                block_rewards = self.episode_rewards[-block_length:]
                block_lengths = self.episode_lengths[-block_length:]
                block_avg_r = sum(block_rewards) / len(block_rewards)
                block_avg_len = sum(block_lengths) / len(block_lengths)
                avg_succ_len = sum(self.block_success_lengths) / len(self.block_success_lengths) if len(self.block_success_lengths) > 0 else 0
                self.success_history.append(self.block_success_count)
                last_loss = self.loss_history[-1] if self.loss_history else float('nan')
                print(f"\n=== Episodes {ep-(block_length-1):4d} {ep-1:4d} summary ===")
                print(f"Avg Reward: {block_avg_r:.2f} | "f"Loss {last_loss:.4f} | "f"Avg Episode Length: {block_avg_len:.2f} | "f"Avg Successful Episode Length: {avg_succ_len:.2f}\n")
                self.block_success_lengths = []
                self.block_success_count = 0

            print(f"Episode {ep:4d}: Reward={total_reward:.2f}, Length={step}, Epsilon={self.epsilon:.4f}, Termination Reason={self.env.termination_reason}, Rocks Collected={self.env.robot_holding}")

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(self.policy_net.state_dict(), CHECKPOINT_PATH)
        print(f"Saved policy network weights to {CHECKPOINT_PATH}")

        num_blocks = len(self.episode_lengths) // block_length
        block_indices = list(range(1, len(self.success_history) + 1))
        block_success_percentages = [ (cnt / block_length) * 100 for cnt in self.success_history ]
        block_avg_lengths = [ sum(self.episode_lengths[i*block_length:(i+1)*block_length]) / block_length for i in range(num_blocks)]   
        
        print(f"Overall Avg Reward: {sum(self.episode_rewards) / len(self.episode_rewards):.2f}")
        print(f"Overall Avg Steps: {sum(self.episode_lengths) / len(self.episode_lengths):.2f}")
        print(f"Overall Avg Steps to Success: {sum(self.steps_to_success) / len(self.steps_to_success) if len(self.steps_to_success) > 0 else 0}")

        print("Overall Cliff Rate: ", self.cliff_fall_count / num_episodes)
        print("Overall Success Rate: ", self.success_count / num_episodes)
        print("Overall Failed Rate: ", (num_episodes - self.success_count - self.cliff_fall_count) / num_episodes)

        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel('Optimization step')
        plt.ylabel('TD-error loss')
        plt.title('DQN Training Loss Curve')
        plt.show()

        plt.figure()
        plt.bar(block_indices, block_success_percentages)
        plt.xlabel("Block Number")
        plt.ylabel("Success Rate (%)")
        plt.title("Block Success Rate (%) Over Blocks")
        plt.xticks(block_indices)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.bar(block_indices, block_avg_lengths)
        plt.xlabel("Block Number")
        plt.ylabel("Average Episode Length")
        plt.title("Average Episode Length per Block")
        plt.xticks(block_indices)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    config = {"size":5, "rocks":[[1,2],[3,3],[2,4]],
              "transmitter_stations":[[4,4]], "cliffs":[[2,3],[1,1]],
              "uphills":[[0,4],[2,0]], "downhills":[[3,0],[0,2]],
              "battery_stations":[[4,2]]}
    env = MarsEnv(config)
    agent = DeepQAgent(env, display=params["display"])
    agent.train()
