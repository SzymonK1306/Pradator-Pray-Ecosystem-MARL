import random
from random import choice

class Agent():
    def __init__(self, id, role, position):
        self.id = id    # TODO think about guid - unique in whole env
        self.role = role
        self.position = position

        self.health = 1

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def get_random_action(self):
        return random.choice([0, 1, 2, 3, 4])   # Actions: 0=stay, 1=up, 2=down, 3=left, 4=right

    def add_health(self, health_gained):
        self.health += health_gained

