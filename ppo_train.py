import sys
import unittest
from collections import deque

import random
import csv
import torch
import torch.optim as optim

from agent import Agent
from actor_critic_model import ActorCriticModel
from env_type1 import PredatorPreyEnv



def batchify(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Wrapping the environment - Can be added in the future

def update_weights_ppo(agent_replay_buffer, agent_policy_model, agent_optimizer, device, clip_epsilon=0.2, gamma=0.99, lam=0.95, value_coeff=0.5, entropy_coeff=0.01):
    # Sample a batch from the replay buffer
    batch = random.sample(agent_replay_buffer, BUFFER_SIZE)

    mini_batches = batchify(batch, BATCH_SIZE)
    for minibatch in mini_batches:

        # Initialize containers for advantages and returns
        advantages = []
        returns = []

        # Process each sample in the batch sequentially
        for obs_mn, action_mn, reward_mn, done_mn, next_obs_mn, hidden_state_mn, next_hidden_state_mn in minibatch:
            # Convert data to tensors
            obs = torch.tensor(obs_mn, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dim
            next_obs = torch.tensor(next_obs_mn, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.tensor(action_mn, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dim
            reward = torch.tensor(reward_mn, dtype=torch.float32, device=device).unsqueeze(0)
            done = torch.tensor(done_mn, dtype=torch.float32, device=device).unsqueeze(0)

            # Compute the value and advantage for this sample
            with torch.no_grad():
                # Get the value and next value from the policy model
                _, value, _ = agent_policy_model(obs, hidden_state_mn)
                _, next_value, _ = agent_policy_model(next_obs, next_hidden_state_mn)

                # Compute advantage (Generalized Advantage Estimation)
                delta = reward + gamma * (1 - done) * next_value.squeeze() - value.squeeze()
                advantage = delta  # Since we're not accumulating over timesteps here
                advantages.append(advantage)

                # Compute the return (advantage + value)
                returns.append(advantage + value.squeeze())

        # Stack all advantages and returns
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)

        # Sequentially optimize the policy for each sample in the batch
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for i, (obs_mn, action_mn, hidden_state_mn) in enumerate(zip(
                [b[0] for b in minibatch],  # obs
                [b[1] for b in minibatch],  # actions
                [b[5] for b in minibatch]   # hidden states
        )):
            # Convert data to tensors
            obs = torch.tensor(obs_mn, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.tensor(action_mn, dtype=torch.long, device=device).unsqueeze(0)

            # Compute the policy and value outputs
            action_probs, value, _ = agent_policy_model(obs, hidden_state_mn)
            action_log_probs = torch.log_softmax(action_probs, dim=-1)
            selected_action_log_prob = torch.gather(action_log_probs,1, action.unsqueeze(1)).squeeze()

            # Compute the ratio
            with torch.no_grad():
                old_action_probs, _, _ = agent_policy_model(obs, hidden_state_mn)
                old_action_log_probs = torch.log_softmax(old_action_probs, dim=-1).gather(1, action.unsqueeze(1)).squeeze()
            ratio = torch.exp(selected_action_log_prob - old_action_log_probs)

            # PPO loss components
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * advantages[i], clipped_ratio * advantages[i])
            value_loss = value_coeff * (returns[i] - value.squeeze()).pow(2)
            entropy_loss = -entropy_coeff * (action_probs * action_log_probs).sum(dim=-1).mean()

            # Accumulate losses
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)

        # Total loss
        loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean() + torch.stack(entropy_losses).mean()

        # Optimize the policy model
        agent_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_policy_model.parameters(), 1.0)
        agent_optimizer.step()

    agent_replay_buffer.clear()


def env_creator():
    env = PredatorPreyEnv((600, 600), 1000, 1000, 1000, 5, 1.0)
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

    csv_file = 'output_ENV_1_PPO.csv'
    data = []

    predator_replay_buffer = deque()
    prey_replay_buffer = deque()

    # Models
    predator_policy_model = ActorCriticModel((4, 51, 51), 4).to(device)
    prey_policy_model = ActorCriticModel((4, 51, 51), 4).to(device)

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
                action_values, _, new_hidden_state = predator_policy_model(obs_tensor, hidden_state)
            else:
                action_values, _, new_hidden_state = prey_policy_model(obs_tensor, hidden_state)

            if random.random() < EPSILON:  # Exploration
                actions[agent.id] = torch.tensor(random.randint(0, 3), device=device)  # Assuming action space is [0, 1, 2, 3]
            else:  # Exploitation
                actions[agent.id] = torch.argmax(action_values)
            new_hidden_states[agent.id] = new_hidden_state

        new_obs, rewards, dones = env.step(actions)

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

        if len(predator_replay_buffer) >= BUFFER_SIZE:
            # Sample a minibatch and train (same as before)
            update_weights_ppo(predator_replay_buffer, predator_policy_model, predator_optimizer, device)
        if len(prey_replay_buffer) >= BUFFER_SIZE:
            # Sample a minibatch and train (same as before)
            update_weights_ppo(prey_replay_buffer, prey_policy_model, prey_optimizer, device)


        num_predators = len([a for a in env.agents if "predator" in a.role])
        num_preys = len([a for a in env.agents if "prey" in a.role])
        data.append([i, num_predators, num_preys])

        obs = new_obs
        hidden_state = new_hidden_states
        print(i, num_predators, num_preys)
        with open(csv_file, mode='a', newline='') as file:  # Open in append mode
            writer = csv.writer(file)
            writer.writerow([i, num_predators, num_preys])
    torch.save(predator_policy_model.state_dict(), "ppo_predator_policy_model.pth")

    torch.save(prey_policy_model.state_dict(), "ppo_prey_policy_model.pth")


