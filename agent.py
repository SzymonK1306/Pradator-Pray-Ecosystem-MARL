import random
from random import choice
from model import DDQNLSTM

class Agent():
    def __init__(self, id, role, position):
        self.id = id    # TODO think about guid - unique in whole env
        self.role = role
        self.position = position

        self.DDQN_model = DDQNLSTM((4, 11, 11), 4)

        if role == 'predator':
            self.health = random.uniform(0.5, 1)
        else:
            self.health = 1

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def get_random_action(self):
        return random.choice([1, 2, 3, 4])   # Actions: 1=up, 2=down, 3=left, 4=right

    def get_DDQN_action(self, obs_tensor, hidden_state):
        q_values, new_hidden_state = self.DDQN_model(obs_tensor, hidden_state)
        action = q_values.argmax().item()
        return action, new_hidden_state

    def add_health(self, health_gained):
        self.health += health_gained

