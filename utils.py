from enum import Enum
import torch
from collections import namedtuple

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    COLLECT = 4
    RECHARGE = 5
    TRANSMIT = 6

class Entities(Enum):
    EMPTY = 0
    EDGE = 1
    ROCK = 2
    UPHILL = 3
    DOWNHILL = 4
    TRANSMITTER_STATION = 5
    BATTERY_STATION = 6
    CLIFF = 7
    ROBOT = 8

class Robot():
    def __init__(self):
        self.position = [0, 0]
        self.battery = 100
        self.holding_rock_count = 0
        self.action = Actions.RIGHT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))