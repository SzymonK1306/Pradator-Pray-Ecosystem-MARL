import sys
import unittest
from collections import deque

import csv
import torch
import torch.optim as optim

from model import DDQNLSTM
from env_type1 import PredatorPreyEnv

def env_creator():
    env = PredatorPreyEnv((600, 600), 1000, 1000, 1000, 5, 1.0)
    return env


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BUFFER_SIZE = 64
    BATCH_SIZE = 64
    EPSILON = 0.1
    UPDATE_FREQ = 50
    GAMMA = 0.99
    LEARNING_RATE = 0.0001

    env = env_creator()
    obs = env.reset()

    csv_file = 'Eval_output_ENV_1_more_hunger_ceil_more_reward_bigger_observation.csv'
    data = []

    predator_replay_buffer = deque()
    prey_replay_buffer = deque()

    # Models
    predator_policy_model = DDQNLSTM((4, 51, 51), 4).to(device)
    prey_policy_model = DDQNLSTM((4, 51, 51), 4).to(device)

    predator_policy_model.load_state_dict(torch.load('predator_policy_model.pth'))
    predator_policy_model.eval()

    prey_policy_model.load_state_dict(torch.load('prey_policy_model.pth'))
    prey_policy_model.eval()

    # Optimizers
    predator_optimizer = optim.Adam(predator_policy_model.parameters(), lr=LEARNING_RATE)
    prey_optimizer = optim.Adam(prey_policy_model.parameters(), lr=LEARNING_RATE)

    hidden_states = {agent.id: None for agent in env.agents}
    new_hidden_states = {agent.id: None for agent in env.agents}

    for i in range(20000):
        actions = {}
        for agent in env.agents:
            obs_tensor = torch.tensor(obs[agent.id], dtype=torch.float32).unsqueeze(0).to(device)
            if agent.id not in hidden_states.keys():
                hidden_state = None
                hidden_states[agent.id] = None
            else:
                hidden_state = hidden_states[agent.id]
            if agent.role == 'predator':
                action_values, new_hidden_state = predator_policy_model(obs_tensor, hidden_state)
            else:
                action_values, new_hidden_state = prey_policy_model(obs_tensor, hidden_state)

            actions[agent.id] = torch.argmax(action_values)
            new_hidden_states[agent.id] = new_hidden_state

        new_obs, rewards, dones = env.step(actions)

        num_predators = len([a for a in env.agents if "predator" in a.role])
        num_preys = len([a for a in env.agents if "prey" in a.role])
        data.append([i, num_predators, num_preys])

        obs = new_obs
        hidden_state = new_hidden_states
        print(i, num_predators, num_preys)
        with open(csv_file, mode='a', newline='') as file:  # Open in append mode
            writer = csv.writer(file)
            writer.writerow([i, num_predators, num_preys])


