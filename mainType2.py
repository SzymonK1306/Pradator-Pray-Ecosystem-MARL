import torch
import torch.optim as optim
import csv
import random
import matplotlib.pyplot as plt
from collections import deque
from predator_prey_env import PredatorPreyEnvType2
from model import DDQNLSTM


# Function to update weights
def update_weights(agent_replay_buffer, agent_policy_model, agent_target_model, agent_optimizer, device='cpu'):
    if len(agent_replay_buffer) < BUFFER_SIZE:
        return
    batch = random.sample(agent_replay_buffer, BATCH_SIZE)

    observations, actions, rewards, dones, next_observations, hidden_states, next_hidden_states = zip(*batch)

    observations = torch.tensor(observations, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    next_observations = torch.tensor(next_observations, dtype=torch.float32).to(device)

    q_values, _ = agent_policy_model(observations, hidden_states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_q_values, _ = agent_target_model(next_observations, next_hidden_states)
        next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + GAMMA * (1 - dones) * next_q_values

    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
    agent_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent_policy_model.parameters(), 1.0)
    agent_optimizer.step()

    agent_target_model.load_state_dict(agent_policy_model.state_dict())


# Hyperparameters
BUFFER_SIZE = 64
BATCH_SIZE = 64
EPSILON = 0.1
UPDATE_FREQ = 50
GAMMA = 0.99
LEARNING_RATE = 0.0001
USE_RANDOM_ACTIONS = True  # Set to False to use policy actions
EPOCHS = 75
# Initialize environment
env = PredatorPreyEnvType2()
obs = env.reset()

# File to store results
csv_file = 'output_ENV_2.csv'
data = []

# Replay buffers
predator_replay_buffer = deque()
prey_replay_buffer = deque()

# Models
predator_policy_model = DDQNLSTM((4, 11, 11), 4).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
predator_target_model = DDQNLSTM((4, 11, 11), 4).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
prey_policy_model = DDQNLSTM((4, 11, 11), 4).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
prey_target_model = DDQNLSTM((4, 11, 11), 4).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Optimizers
predator_optimizer = optim.Adam(predator_policy_model.parameters(), lr=LEARNING_RATE)
prey_optimizer = optim.Adam(prey_policy_model.parameters(), lr=LEARNING_RATE)

hidden_states = {agent.id: None for agent in env.agents}
new_hidden_states = {agent.id: None for agent in env.agents}

epochs = []
predator_counts = []
prey_counts = []

for i in range(EPOCHS):
    actions = {}
    frozen_agents = set()
    env.ensure_population()
    print(f"Iteration {i}")

    for agent in env.agents:
        if agent in frozen_agents:
            continue

        # obs_tensor = torch.tensor(obs[agent.id], dtype=torch.float32).unsqueeze(0)
        # hidden_state = hidden_states.get(agent.id, None)

        if USE_RANDOM_ACTIONS:
            actions[agent.id] = random.randint(0, 3)
            new_hidden_state = None
        else:
            obs_tensor = torch.tensor(obs[agent.id], dtype=torch.float32).unsqueeze(0)
            hidden_state = hidden_states.get(agent.id, None)
            if agent.role == 'predator':
                action_values, new_hidden_state = predator_policy_model(obs_tensor, hidden_state)
            else:
                action_values, new_hidden_state = prey_policy_model(obs_tensor, hidden_state)

            if random.random() < EPSILON:
                actions[agent.id] = random.randint(0, 3)
            else:
                actions[agent.id] = torch.argmax(action_values).item()

        new_hidden_states[agent.id] = new_hidden_state

    new_obs, rewards, dones = env.step(actions)
    num_matings = rewards.get('mating_count', 0)
    print(f"Number of matings this iteration: {num_matings}")

    num_predators = len([a for a in env.agents if a.role == "predator"])
    num_preys = len([a for a in env.agents if a.role == "prey"])
    print(f"Predators: {num_predators}, Preys: {num_preys}")

    epochs.append(i)
    predator_counts.append(num_predators)
    prey_counts.append(num_preys)

    if num_predators + num_preys < len(env.agents):
        print("Reproduction occurred!")

    # for agent_id in actions.keys():
    #     if dones[agent_id]:
    #         new_obs_to_save = torch.zeros_like(torch.tensor(obs[agent_id], dtype=torch.float32))
    #     else:
    #         new_obs_to_save = new_obs[agent_id]
    #
    #     experience = (
    #         obs[agent_id],
    #         actions[agent_id],
    #         rewards[agent_id],
    #         dones[agent_id],
    #         new_obs_to_save,
    #         hidden_states[agent_id],
    #         new_hidden_states[agent_id]
    #     )
    #
    #     if agent_id.startswith('pr'):
    #         predator_replay_buffer.append(experience)
    #     else:
    #         prey_replay_buffer.append(experience)

    if not USE_RANDOM_ACTIONS:
        pass
        # update_weights(predator_replay_buffer, predator_policy_model, predator_target_model, predator_optimizer)
    if not USE_RANDOM_ACTIONS:
        for agent_id in actions.keys():
            if dones[agent_id]:
                new_obs_to_save = torch.zeros_like(torch.tensor(obs[agent_id], dtype=torch.float32))
            else:
                new_obs_to_save = new_obs[agent_id]

            experience = (
                obs[agent_id],
                actions[agent_id],
                rewards[agent_id],
                dones[agent_id],
                new_obs_to_save,
                hidden_states[agent_id],
                new_hidden_states[agent_id]
            )

            if agent_id.startswith('pr'):
                predator_replay_buffer.append(experience)
            else:
                prey_replay_buffer.append(experience)

        update_weights(prey_replay_buffer, prey_policy_model, prey_target_model, prey_optimizer)

    data.append([i, num_predators, num_preys])

    obs = new_obs
    hidden_states = new_hidden_states

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i, num_predators, num_preys])

plt.figure(figsize=(10, 5))
plt.plot(epochs, predator_counts, label='Predators')
plt.plot(epochs, prey_counts, label='Preys')
plt.xlabel('Epoch')
plt.ylabel('Population Count')
plt.title('Population Dynamics Over Time')
plt.legend()
plt.savefig("population_plot.png")
plt.show()

torch.save(predator_target_model.state_dict(), "predator_target_model_Type2.pth")
torch.save(predator_policy_model.state_dict(), "predator_policy_model_Type2.pth")
torch.save(prey_target_model.state_dict(), "prey_target_model_Type2.pth")
torch.save(prey_policy_model.state_dict(), "prey_policy_model_Type2.pth")
