import random

class AgentType3():
    def __init__(self, id, role, position):
        self.id = id
        self.role = role
        self.position = position

        if role == 'predator':
            self.health = random.uniform(0.5, 1)
            self.speed = random.uniform(0.5, 1.5)
            self.attack = random.uniform(0.5, 1.5)
            self.resilience = 0
        else:
            self.health = 1
            self.speed = random.uniform(0.5, 1.5)
            self.resilience = random.uniform(0.5, 1.5)
            self.attack = 0

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def get_random_action(self):
        # Actions: 1 = up, 2 = down, 3 = left, 4 = right
        return random.choice([1, 2, 3, 4])

    def add_health(self, health_gained):
        self.health += health_gained