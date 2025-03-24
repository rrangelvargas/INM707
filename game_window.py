import pygame
import os
from utils import Actions

class GameWindow:
    def __init__(self, window_size=800):
        pygame.init()
        self.window_size = window_size
        self.sidebar_width = 200
        self.display = pygame.display.set_mode((self.window_size + self.sidebar_width, self.window_size))
        pygame.display.set_caption('Mars Space Exploration')
        
        self.BLACK = (0, 0, 0)
        self.LIGHT_GRAY = (240, 240, 240)
        self.RED = (255, 0, 0)
        self.GRID_COLOR = (200, 200, 200)
        
        self.info_font = pygame.font.Font(None, 32)
    
    def draw_sidebar(self, episode, step, epsilon=None, temperature=None, policy=None):
        sidebar_rect = pygame.Rect(0, 0, self.sidebar_width, self.window_size)
        pygame.draw.rect(self.display, self.LIGHT_GRAY, sidebar_rect)
        pygame.draw.line(self.display, self.BLACK, (self.sidebar_width, 0), (self.sidebar_width, self.window_size), 2)
        
        episode_text = self.info_font.render(f"Episode: {episode}", True, self.BLACK)
        self.display.blit(episode_text, (20, 30))
        
        step_text = self.info_font.render(f"Step: {step}", True, self.BLACK)
        self.display.blit(step_text, (20, 70))
        
        if policy is not None:
            policy_text = self.info_font.render(f"Policy: {policy}", True, self.BLACK)
            self.display.blit(policy_text, (20, 190))
        
        if epsilon is not None and policy == "epsilon_greedy":
            epsilon_text = self.info_font.render(f"Epsilon: {epsilon:.4f}", True, self.RED)
            self.display.blit(epsilon_text, (20, 110))
        
        if temperature is not None and policy == "softmax":
            temp_text = self.info_font.render(f"Temp: {temperature:.4f}", True, self.RED)
            self.display.blit(temp_text, (20, 150))
    
    def draw_grid(self, grid_size):
        for x in range(self.sidebar_width, self.window_size + self.sidebar_width, grid_size):
            pygame.draw.line(self.display, self.BLACK, (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, grid_size):
            pygame.draw.line(self.display, self.BLACK, (self.sidebar_width, y), (self.window_size + self.sidebar_width, y))
    
    def process_image(self, filename, scale_factor, grid_size):
        path = os.path.join(os.getcwd(), "images", filename)
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (int(grid_size * scale_factor), int(grid_size * scale_factor)))
        image = image.convert_alpha()
        return image
    
    def load_images(self, grid_size):
        _robot = self.process_image("robot.png", 0.6, grid_size)
        _rock = self.process_image("rock.png", 0.6, grid_size)
        _transmiter = self.process_image("antenna.png", 0.6, grid_size)
        _cliff = self.process_image("cliff.png", 0.6, grid_size)
        _uphill = self.process_image("uphill.png", 0.6, grid_size)
        _downhill = self.process_image("downhill.png", 0.6, grid_size)
        _battery = self.process_image("battery.png", 0.6, grid_size)
        return _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery
    
    def render_images(self, robot, rocks, transmiter_stations, cliffs, uphills, downhills, batery_stations, grid_size):
        _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery = self.load_images(grid_size)
        self.blit_objects(_robot, [robot.position], grid_size)
        self.blit_objects(_rock, rocks, grid_size)
        self.blit_objects(_transmiter, transmiter_stations, grid_size)
        self.blit_objects(_cliff, cliffs, grid_size)
        self.blit_objects(_uphill, uphills, grid_size)
        self.blit_objects(_downhill, downhills, grid_size)
        self.blit_objects(_battery, batery_stations, grid_size)
    
    def blit_objects(self, image, objects, grid_size, offset=0.2):
        for obj in objects:
            pos = (self.sidebar_width + obj[0] * grid_size + grid_size * offset, 
                   obj[1] * grid_size + grid_size * offset)
            self.display.blit(image, pos)
