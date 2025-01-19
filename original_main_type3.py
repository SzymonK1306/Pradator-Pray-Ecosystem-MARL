import math
import sys
import unittest
from collections import deque

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers
import numpy as np
import random
import csv
import torch
import torch.optim as optim

from agent import Agent
from model import DDQNLSTM

from env_typ3 import PredatorPreyEnvType3

def batchify(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def update_weights(agent_replay_buffer, agent_policy_model, agent_target_model, agent_optimizer, device='cpu'):
    batch = random.sample(agent_replay_buffer, BUFFER_SIZE)

    mini_batches = batchify(batch, BATCH_SIZE)
    for minibatch in mini_batches:
        # Prepare batches for training
        # obs_batch = torch.tensor([exp[0] for exp in minibatch], dtype=torch.float32)
        # action_batch = torch.tensor([exp[1] for exp in minibatch], dtype=torch.long)
        # reward_batch = torch.tensor([exp[2] for exp in minibatch], dtype=torch.float32)
        # done_batch = torch.tensor([exp[3] for exp in minibatch], dtype=torch.float32)
        # next_obs_batch = torch.empty(len(minibatch), 4, 11, 11, dtype=torch.float32)  # Allocate an empty tensor
        # for ii, exp in enumerate(minibatch):
        #     next_obs_batch[ii] = torch.tensor(exp[4])
        # # next_obs_batch = torch.tensor([exp[4] for exp in minibatch], dtype=torch.float32)
        # hidden_state_batch = [exp[5] for exp in minibatch]
        # new_hidden_state_batch = [exp[6] for exp in minibatch]

        # Compute target Q-values and optimize
        q_values_batch = []
        target_q_values = []
        for obs_mn, action_mn, reward_mn, done_mn, next_obs_mn, hidden_state_mn, next_hidden_state_mn in minibatch:
            with torch.no_grad():
                next_obs = torch.tensor(next_obs_mn, dtype=torch.float32).unsqueeze(0).to(device)
                next_action = torch.argmax(agent_policy_model(next_obs, next_hidden_state_mn)[0])
                target_q_value = reward_mn + GAMMA * (1 - done_mn) * \
                                 agent_target_model(next_obs, next_hidden_state_mn)[0].squeeze(0)[next_action]
                target_q_values.append(target_q_value)
            q_values, _ = agent_policy_model(torch.tensor(obs_mn, dtype=torch.float32, device=device).unsqueeze(0), hidden_state_mn)
            q_value = q_values.gather(1, action_mn.view(1, 1)).squeeze()
            q_values_batch.append(q_value)
        target_q_values = torch.stack(target_q_values)

        q_values_batch = torch.stack((q_values_batch))
        # Compute current Q-values and loss
        # if all(x is None for x in hidden_state_batch):
        #     q_values, _ = agent_policy_model(obs_batch)
        # else:
        #     q_values, _ = agent_policy_model(obs_batch, hidden_state_batch)
        # q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        loss = torch.nn.functional.mse_loss(q_values_batch, target_q_values)

        # Optimize the shared network
        agent_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_policy_model.parameters(), 1.0)
        agent_optimizer.step()

    if i % UPDATE_FREQ == 0:
        agent_target_model.load_state_dict(agent_policy_model.state_dict())

        torch.save(predator_target_model.state_dict(), "predator_target_model.pth")
        torch.save(predator_policy_model.state_dict(), "predator_policy_model.pth")
        torch.save(prey_target_model.state_dict(), "prey_target_model.pth")
        torch.save(prey_policy_model.state_dict(), "prey_policy_model.pth")

    agent_replay_buffer.clear()
# Wrapping the environment - Can be added in the future

def env_creator():
    env = PredatorPreyEnvType3((600, 600), 1000, 1000, 10000, 5, 1.0)
    return env

RUN_TESTS_BEFORE = False

def run_tests():
    print("Running tests...")
    
    test_suite = unittest.defaultTestLoader.discover(start_dir='.', pattern='test_*.py')
    test_runner = unittest.TextTestRunner()
    result = test_runner.run(test_suite)

    if not result.wasSuccessful():
        print("Tests failed! The program will be terminated...")
        sys.exit(1)
    else:
        print("All tests passed! Proceeding to main program...")

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if RUN_TESTS_BEFORE:
        run_tests() 
    else:
        print("WARNING: running without tests...")    

    # Hyperparameters
    BUFFER_SIZE = 64
    BATCH_SIZE = 64
    EPSILON = 0.1
    UPDATE_FREQ = 50
    GAMMA = 0.99
    LEARNING_RATE = 0.0001

    env = env_creator()
    obs = env.reset()
    # env.render()

    csv_file = 'output_ENV_1_more_hunger_ceil.csv'
    data = []

    predator_replay_buffer = deque()
    prey_replay_buffer = deque()

    # Models
    predator_policy_model = DDQNLSTM((7, 11, 11), 4).to(device)
    predator_target_model = DDQNLSTM((7, 11, 11), 4).to(device)
    prey_policy_model = DDQNLSTM((7, 11, 11), 4).to(device)
    prey_target_model = DDQNLSTM((7, 11, 11), 4).to(device)

    # Optimizers
    predator_optimizer = optim.Adam(predator_policy_model.parameters(), lr=LEARNING_RATE)
    prey_optimizer = optim.Adam(prey_policy_model.parameters(), lr=LEARNING_RATE)

    hidden_states = {agent.id: None for agent in env.agents}
    new_hidden_states = {agent.id: None for agent in env.agents}

    for i in range(20000):
        actions = {}
        # actions = {agent.id: random.randint(0, 4) for agent in env.agents}
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

            if random.random() < EPSILON:  # Exploration
                actions[agent.id] = torch.tensor(random.randint(0, 3), device=device)  # Assuming action space is [0, 1, 2, 3]
            else:  # Exploitation
                actions[agent.id] = torch.argmax(action_values)
            new_hidden_states[agent.id] = new_hidden_state

        new_obs, rewards, dones = env.step(actions)

        # experiences["observations"].update(obs)
        # experiences["actions"].update(actions)
        # experiences["rewards"].update(rewards)
        # experiences["dones"].update(dones)
        for agent_id in actions.keys():
            if dones[agent_id]:
                new_obs_to_save = torch.zeros_like(torch.tensor(obs[agent_id], dtype=torch.float32)).to(device)  # Placeholder
            else:
                new_obs_to_save = new_obs[agent_id]
            experience = (
                obs[agent_id],  # Current observation
                actions[agent_id],  # Action taken
                rewards[agent_id],  # Reward received
                dones[agent_id],  # Done flag
                new_obs_to_save,  # Next observation
                hidden_states[agent_id],  # Current hidden state
                new_hidden_states[agent_id]
            )
            if agent_id[:2] == 'pr':
                predator_replay_buffer.append(experience)
            else:
                prey_replay_buffer.append(experience)

        # env.generate_new_agents()
        if len(predator_replay_buffer) >= BUFFER_SIZE:
            # Sample a minibatch and train (same as before)
            update_weights(predator_replay_buffer, predator_policy_model, predator_target_model, predator_optimizer, device)
        if len(prey_replay_buffer) >= BUFFER_SIZE:
            # Sample a minibatch and train (same as before)
            update_weights(prey_replay_buffer, prey_policy_model, prey_target_model, prey_optimizer, device)


        num_predators = len([a for a in env.agents if "predator" in a.role])
        num_preys = len([a for a in env.agents if "prey" in a.role])
        data.append([i, num_predators, num_preys])

        obs = new_obs
        hidden_state = new_hidden_states
        print(i, num_predators, num_preys)
        with open(csv_file, mode='a', newline='') as file:  # Open in append mode
            writer = csv.writer(file)
            writer.writerow([i, num_predators, num_preys])
    torch.save(predator_target_model.state_dict(), "predator_target_model.pth")
    torch.save(predator_policy_model.state_dict(), "predator_policy_model.pth")

    torch.save(prey_target_model.state_dict(), "prey_target_model.pth")
    torch.save(prey_policy_model.state_dict(), "prey_policy_model.pth")
        # env.render()

    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #
    #     # Loop through each item in data
    #     for row in data:
    #         # Extract the last three elements
    #         last_three = row[-3:]
    #
    #         # Write them to the CSV
    #         writer.writerow(last_three)


