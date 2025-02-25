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

# 0 (0,0)
# 1 (1,0)
# 2 (2,0)
# 3 (3,0)
# 4 (4,0)
# 5 (0,1)
# 6 (1,1)
# 7 (2,1)
# 8 (3,1)
# 9 (4,1)
# 10 (0,2)
# 11 (1,2)
# 12 (2,2)
# 13 (3,2)
# 14 (4,2)
# 15 (0,3)
# 16 (1,3)
# 17 (2,3)
# 18 (3,3)
# 19 (4,3)
# 20 (0,4)
# 21 (1,4)
# 22 (2,4)
# 23 (3,4)
# 24 (4,4)

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


    def load_images(self):
        _robot = pygame.image.load(f"{os.getcwd()}/images/robot.png")
        _robot = pygame.transform.scale(_robot, (int(self.grid_size) * 0.8, int(self.grid_size) * 0.8))
        _robot.convert()

        _rock = pygame.image.load(f"{os.getcwd()}/images/rock.png")
        _rock = pygame.transform.scale(_rock, (int(self.grid_size) * 0.6, int(self.grid_size) * 0.6))
        _rock.convert()

        _transmiter = pygame.image.load(f"{os.getcwd()}/images/antenna.png")
        _transmiter = pygame.transform.scale(_transmiter, (int(self.grid_size) * 0.6, int(self.grid_size) * 0.6))
        _transmiter.convert()

        _cliff = pygame.image.load(f"{os.getcwd()}/images/cliff.png")
        _cliff = pygame.transform.scale(_cliff, (int(self.grid_size) * 0.6, int(self.grid_size) * 0.6))
        _cliff.convert()

        _uphill = pygame.image.load(f"{os.getcwd()}/images/uphill.png")
        _uphill = pygame.transform.scale(_uphill, (int(self.grid_size) * 0.6, int(self.grid_size) * 0.6))
        _uphill.convert()

        _downhill = pygame.image.load(f"{os.getcwd()}/images/downhill.png")
        _downhill = pygame.transform.scale(_downhill, (int(self.grid_size) * 0.6, int(self.grid_size) * 0.6))
        _downhill.convert()

        _battery = pygame.image.load(f"{os.getcwd()}/images/battery.png")
        _battery = pygame.transform.scale(_battery, (int(self.grid_size) * 0.6, int(self.grid_size) * 0.6))
        _battery.convert()

        return _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery
    
    def render_images(self, _rock, _transmiter, _cliff, _uphill, _downhill, _battery):
            for rock in self.rocks:
                rock_pos = (rock[0] * self.grid_size + self.grid_size * 0.2,
                            rock[1] * self.grid_size + self.grid_size * 0.2)
                self.display.blit(_rock, rock_pos)

            for transmiter in self.transmiter_stations:
                transmiter_pos = (transmiter[0] * self.grid_size + self.grid_size * 0.2,
                                  transmiter[1] * self.grid_size + self.grid_size * 0.2)
                self.display.blit(_transmiter, transmiter_pos)

            for cliff in self.cliffs:
                cliff_pos = (cliff[0] * self.grid_size + self.grid_size * 0.2,
                             cliff[1] * self.grid_size + self.grid_size * 0.2)
                self.display.blit(_cliff, cliff_pos)
            
            for uphill in self.uphills:
                uphill_pos = (uphill[0] * self.grid_size + self.grid_size * 0.2,
                              uphill[1] * self.grid_size + self.grid_size * 0.2)
                self.display.blit(_uphill, uphill_pos)

            for downhill in self.downhills:
                downhill_pos = (downhill[0] * self.grid_size + self.grid_size * 0.2,
                                downhill[1] * self.grid_size + self.grid_size * 0.2)
                self.display.blit(_downhill, downhill_pos)

            for battery in self.batery_stations:
                battery_pos = (battery[0] * self.grid_size + self.grid_size * 0.2,
                               battery[1] * self.grid_size + self.grid_size * 0.2)
                self.display.blit(_battery, battery_pos)
                

    def render(self):
        pygame.init()
        pygame.display.init()
        self.display = pygame.display.set_mode((self.size, self.size))
        self.clock = pygame.time.Clock()
        screen = pygame.display.set_mode((self.window, self.window))
        pygame.display.set_caption('Mars Space Exploration')
        self.grid_size = int(self.window/self.size)
        self.grid_colour = (200, 200, 200)
        self.line_colour = (100, 100, 100)

        _robot, _rock, _transmiter, _cliff, _uphill, _downhill, _battery = self.load_images()

        screen_on = True
        while screen_on:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            screen.fill(self.grid_colour)
            for x in range(0, self.window, self.grid_size):
                pygame.draw.line(screen, self.line_colour, (x, 0), (x, self.window))
            for y in range(0, self.window, self.grid_size):
                pygame.draw.line(screen, self.line_colour, (0, y), (self.window, y))

            robot_pos = (self.robot.position[0] * self.grid_size + self.grid_size * 0.1,
                         self.robot.position[1] * self.grid_size + self.grid_size * 0.1)

            self.display.blit(_robot, robot_pos)

            self.render_images(_rock, _transmiter, _cliff, _uphill, _downhill, _battery)

            pygame.display.flip()

    def update_q_table(self):
        pass


mars_environment = Mars_Environment(5)