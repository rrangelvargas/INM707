import pygame
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import Actions

class GameWindow:
    def __init__(self, window_size=800):
        # Initialize Pygame first
        pygame.init()
        
        self.window_size = window_size
        self.sidebar_width = 200
        # Initialize with extra width for sidebar
        self.display = pygame.display.set_mode((self.window_size + self.sidebar_width, self.window_size))
        pygame.display.set_caption('Mars Space Exploration')
        
        self.BLACK = (0, 0, 0)
        self.LIGHT_GRAY = (240, 240, 240)
        self.RED = (255, 0, 0)
        self.GRID_COLOR = (200, 200, 200)
        
        self.info_font = pygame.font.Font(None, 32)
        
    def draw_sidebar(self, episode, step, epsilon=None, temperature=None, policy=None):
        # Draw sidebar background
        sidebar_rect = pygame.Rect(0, 0, self.sidebar_width, self.window_size)
        pygame.draw.rect(self.display, self.LIGHT_GRAY, sidebar_rect)
        pygame.draw.line(self.display, self.BLACK, (self.sidebar_width, 0), 
                        (self.sidebar_width, self.window_size), 2)
        
        # Draw episode info
        episode_text = self.info_font.render(f"Episode: {episode}", True, self.BLACK)
        self.display.blit(episode_text, (20, 30))
        
        # Draw step info
        step_text = self.info_font.render(f"Step: {step}", True, self.BLACK)
        self.display.blit(step_text, (20, 70))

        # Draw policy info if provided
        if policy is not None:
            policy_text = self.info_font.render(f"Policy: {policy}", True, self.BLACK)
            self.display.blit(policy_text, (20, 190))

        # Draw epsilon info if provided
        if epsilon is not None and policy == "episilon_greedy":
            epsilon_text = self.info_font.render(f"Epsilon: {epsilon:.4f}", True, self.RED)
            self.display.blit(epsilon_text, (20, 110))

        # Draw temperature info if provided
        if temperature is not None and policy == "softmax":
            temp_text = self.info_font.render(f"Temp: {temperature:.4f}", True, self.RED)
            self.display.blit(temp_text, (20, 150))

    def draw_grid(self, grid_size):
        # Draw grid lines offset by sidebar width
        for x in range(self.sidebar_width, self.window_size + self.sidebar_width, grid_size):
            pygame.draw.line(self.display, self.BLACK, (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, grid_size):
            pygame.draw.line(self.display, self.BLACK, 
                           (self.sidebar_width, y), 
                           (self.window_size + self.sidebar_width, y))

    def render_game(self, mars_env):            
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption('Mars Space Exploration')
        self.grid_size = int(self.window_size/mars_env.size)

        reward = 0
        steps = []
        episode_numbers = []
        
        # Initialize counters
        goal_reached_count = 0
        cliff_falls_count = 0
        battery_depleted_count = 0
        successful_mission_steps = []  # List to store steps for successful missions

        for episode in range(mars_env.no_episodes):
            print(f"EPISODE NO: {episode+1}")
            epsilon = mars_env.min_epsilon + (mars_env.max_epsilon - mars_env.min_epsilon)*np.exp(-mars_env.epsilon_decay_rate*episode)
            temperature = mars_env.min_temperature + (mars_env.max_temperature - mars_env.min_temperature)*np.exp(-mars_env.temperature_decay_rate*episode)

            mars_env.reset()

            for step in range(mars_env.max_steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Fill only the game area, not the sidebar
                pygame.draw.rect(self.display, self.GRID_COLOR, 
                               (self.sidebar_width, 0, self.window_size, self.window_size))
                
                # Draw grid
                self.draw_grid(self.grid_size)

                # Update sidebar with additional information
                self.draw_sidebar(episode + 1, step + 1, epsilon, temperature, mars_env.policy)

                # Adjust object rendering to account for sidebar offset
                self.render_images(mars_env.robot, mars_env.rocks, mars_env.transmiter_stations, mars_env.cliffs, mars_env.uphills, mars_env.downhills, mars_env.batery_stations)

                pygame.display.flip()

                if mars_env.robot.position in mars_env.cliffs:
                    print("--------------------------------")
                    print(f"STEP NO: {step+1} \n")
                    print(f"Position: {mars_env.robot.position}")
                    print(f"Battery: {mars_env.robot.battery}")
                    print(f"Reward: {reward}")
                    print(f"Epsilon: {epsilon}")
                    print(f"Temperature: {temperature}")
                    print("--------------------------------\n")

                    print("fell off a cliff")
                    cliff_falls_count += 1
                    break

                if mars_env.robot.battery <= 0:
                    print("--------------------------------")
                    print(f"STEP NO: {step+1} \n")
                    print(f"Position: {mars_env.robot.position}")
                    print(f"Battery: {mars_env.robot.battery}")
                    print(f"Reward: {reward}")
                    print(f"Epsilon: {epsilon}")
                    print(f"Temperature: {temperature}")
                    print("--------------------------------")
                    
                    print("ran out of battery")
                    battery_depleted_count += 1
                    break

                if mars_env.robot.action == Actions.TRANSMIT:
                    print("--------------------------------")
                    print(f"STEP NO: {step+1} \n")
                    print(f"Position: {mars_env.robot.position}")
                    print(f"Battery: {mars_env.robot.battery}")
                    print(f"Reward: {reward}")
                    print(f"Epsilon: {epsilon}")
                    print(f"Temperature: {temperature}")
                    print("--------------------------------")

                    if not len(mars_env.rocks):
                        print(f"Goal Reached, all rocks collected! Episode {episode+1} has ended")
                        goal_reached_count += 1
                        successful_mission_steps.append(step + 1)
                        episode_numbers.append(episode)
                        steps.append(step)
                        break

                print("--------------------------------")
                print(f"STEP NO: {step+1} \n")
                print(f"Position: {mars_env.robot.position}")
                print(f"Battery: {mars_env.robot.battery}")
                print(f"Reward: {reward}")
                print(f"Epsilon: {epsilon}")
                print(f"Temperature: {temperature}")

                old_position = mars_env.robot.position
                action = mars_env.choose_action(epsilon, temperature)
                mars_env.update_robot(action)
                reward = mars_env.calculate_reward(old_position)
                mars_env.update_q_table(action, reward, old_position)

                print("--------------------------------")

        # Calculate averages
        avg_steps_successful_mission = sum(successful_mission_steps) / len(successful_mission_steps) if successful_mission_steps else 0
        
        # Display final statistics
        print("\n=== Final Statistics ===")
        print(f"Total Episodes: {mars_env.no_episodes}")
        print(f"Goals Reached: {goal_reached_count} ({(goal_reached_count/mars_env.no_episodes)*100:.2f}%)")
        print(f"Cliff Falls: {cliff_falls_count} ({(cliff_falls_count/mars_env.no_episodes)*100:.2f}%)")
        print(f"Battery Depletions: {battery_depleted_count} ({(battery_depleted_count/mars_env.no_episodes)*100:.2f}%)")
        print(f"Failed Episodes: {mars_env.no_episodes - goal_reached_count - cliff_falls_count - battery_depleted_count} ({((mars_env.no_episodes - goal_reached_count - cliff_falls_count - battery_depleted_count)/mars_env.no_episodes)*100:.2f}%)")
        print(f"Average Steps for Successful Missions: {avg_steps_successful_mission:.2f}")
        print("=====================\n")
        
        # Plot results
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

    def render_images(self, robot, rocks, transmiter_stations, cliffs, uphills, downhills, batery_stations):
        _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery = self.load_images()
        self.blit_objects(_robot, [robot.position])
        self.blit_objects(_rock, rocks)
        self.blit_objects(_transmiter, transmiter_stations)
        self.blit_objects(_cliff, cliffs)
        self.blit_objects(_uphill, uphills)
        self.blit_objects(_downhill, downhills)
        self.blit_objects(_battery, batery_stations)

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
            pos = (self.sidebar_width + obj[0] * self.grid_size + self.grid_size * offset, 
                  obj[1] * self.grid_size + self.grid_size * offset)
            self.display.blit(image, pos)