import pygame
from enum import Enum
import sys
import os

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

class Mars_Environment():
    robot = Robot()
    def __init__(self, size):
        self.size = size
        self.window = 800
        self.display = None
        self.clock = None
        self.q_table = [[0 for _ in range(len(Actions))] for _ in range(size * size)]
        self.rewards = [[0 for _ in range(size * size)] for _ in range(size * size)]
        self.robot = [(0,0)]
        self.rocks = [(1,2), (3,3), (2,4)]
        self.transmiter_stations = [(4,4)]
        self.cliffs = [(2,3), (1,1)]
        self.uphills = [(0,4), (2,0)]
        self.downhills = [(3,0), (0,2)]
        self.batery_stations = [(4,3)]
        self.fill_rewards()
        self.render()

    def reset(self):
        self.robot.position = (0, 0)
        self.robot.action = Actions.RIGHT
        self.robot.battery = 100
        self.robot.holding_rock = False

    def fill_rewards(self):
        for rock in self.rocks:
            self.rewards[rock[0] + self.size*rock[1]][rock[0] + self.size*rock[1]] = 20
        
        for transmiter in self.transmiter_stations:
            self.rewards[transmiter[0] + self.size*transmiter[1]][transmiter[0] + self.size*transmiter[1]] = 50

        for battery in self.batery_stations:
            self.rewards[battery[0] + self.size*battery[1]][battery[0] + self.size*battery[1]] = 20

        for cliff in self.cliffs:
            if cliff[0] > 0:
                self.rewards[cliff[0]-1 + self.size*cliff[1]][cliff[0] + self.size*cliff[1]] = -50
            if cliff[0] < self.size-1:
                self.rewards[cliff[0]+1 + self.size*cliff[1]][cliff[0] + self.size*cliff[1]] = -50
            if cliff[1] > 0:
                self.rewards[cliff[0] + self.size*(cliff[1]-1)][cliff[0] + self.size*cliff[1]] = -50
            if cliff[1] < self.size-1:
                self.rewards[cliff[0] + self.size*(cliff[1]+1)][cliff[0] + self.size*cliff[1]] = -50
        
        for uphill in self.uphills:
            if uphill[0] > 0:
                self.rewards[uphill[0]-1 + self.size*uphill[1]][uphill[0] + self.size*uphill[1]] = -5
            if uphill[0] < self.size-1:
                self.rewards[uphill[0]+1 + self.size*uphill[1]][uphill[0] + self.size*uphill[1]] = -5
            if uphill[1] > 0:
                self.rewards[uphill[0] + self.size*(uphill[1]-1)][uphill[0] + self.size*uphill[1]] = -5                
            if uphill[1] < self.size-1:
                self.rewards[uphill[0] + self.size*(uphill[1]+1)][uphill[0] + self.size*uphill[1]] = -5

        for downhill in self.downhills:
            if downhill[0] > 0:
                self.rewards[downhill[0]-1 + self.size*downhill[1]][downhill[0] + self.size*downhill[1]] = 5
            if downhill[0] < self.size-1:
                self.rewards[downhill[0]+1 + self.size*downhill[1]][downhill[0] + self.size*downhill[1]] = 5
            if downhill[1] > 0:
                self.rewards[downhill[0] + self.size*(downhill[1]-1)][downhill[0] + self.size*downhill[1]] = 5
            if downhill[1] < self.size-1:
                self.rewards[downhill[0] + self.size*(downhill[1]+1)][downhill[0] + self.size*downhill[1]] = 5


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
            pos = (obj[0] * self.grid_size + self.grid_size * offset, obj[1] * self.grid_size + self.grid_size * offset)
            self.display.blit(image, pos)

    def render_images(self, _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery):
        self.blit_objects(_robot, self.robot)
        self.blit_objects(_rock, self.rocks)
        self.blit_objects(_transmiter, self.transmiter_stations)
        self.blit_objects(_cliff, self.cliffs)
        self.blit_objects(_uphill, self.uphills)
        self.blit_objects(_downhill, self.downhills)
        self.blit_objects(_battery, self.batery_stations)

    def render(self):
        pygame.init()
        pygame.display.init()
        self.display = pygame.display.set_mode((self.window, self.window))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Mars Space Exploration')
        self.grid_size = int(self.window/self.size)
        self.grid_colour = (200, 200, 200)
        self.line_colour = (0, 0, 0)
        _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery = self.load_images()

        display_on = True
        while display_on:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.display.fill(self.grid_colour)
            for x in range(0, self.window, self.grid_size):
                pygame.draw.line(self.display, self.line_colour, (x, 0), (x, self.window))
            for y in range(0, self.window, self.grid_size):
                pygame.draw.line(self.display, self.line_colour, (0, y), (self.window, y))

            self.render_images(_robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery)

            pygame.display.flip()

    def update_q_table(self):
        pass

mars_environment = Mars_Environment(5)
