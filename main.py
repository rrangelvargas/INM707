import pygame
from enum import Enum
import sys

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
        self.action = Actions.RIGHT
        self.battery = 100
        self.rock_count = 0

class Mars_Environment():
    robot = Robot()
    def __init__(self, size):
        self.window = 800
        self.display = None
        self.clock = None
        self.q_table = [[0 for _ in range(len(Actions))] for _ in range(size * size)]
        self.rewards = [[0 for _ in range(size * size)] for _ in range(size * size)]
        self.rocks = [(1,2), (3,3), (2,4)]
        self.size = size
        #self.transmiter = []
        #self.cliffs = []
        #self.uphills = []
        #self.downhills = []
        self.fill_rewards()

        print(self.q_table)
        print(self.rewards)

    def reset(self):
        self.robot.position = (0, 0)
        self.robot.action = Actions.RIGHT
        self.robot.battery = 100
        self.robot.holding_rock = False

    def fill_rewards(self):
        for rock in self.rocks:
            self.rewards[rock[0] + self.size*rock[1]][rock[0] + self.size*rock[1]] = 20

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
        robot = pygame.image.load("C:\\Users\\lukep\\OneDrive\\Desktop\\Artificial Intelligence\\INM707\\pygame\\robot.png")
        robot = pygame.transform.scale(robot, (int(self.grid_size) * 0.8, int(self.grid_size) * 0.8))
        robot.convert()

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

            self.display.blit(robot, robot_pos)

            pygame.display.flip()

    def update_q_table(self):
        pass


mars_environment = Mars_Environment(5)
mars_environment.render()
