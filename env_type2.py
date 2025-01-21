
from pettingzoo.utils.env import ParallelEnv
import numpy as np
import random
from agent import Agent
import math
from env_type1 import PredatorPreyEnv


class PredatorPreyEnvType2(PredatorPreyEnv):
    def __init__(self, grid_size=(600, 600), num_predators=2000, num_prey=1000, num_walls=10, predator_scope=5,
                 health_gained=1.0, health_penalty=0.01, mating_scope=15, mating_reward=4, predator_mating_probability=0.003,
                 prey_mating_probability=0.006):
        """
        Initializes the extended environment for Type 2 simulation with mating behavior and environment settings.
        """
        super().__init__(grid_size, num_predators, num_prey, num_walls, predator_scope, health_gained, health_penalty)
        self.mating_scope = mating_scope
        self.mating_reward = mating_reward
        self.predator_mating_probability = predator_mating_probability
        self.prey_mating_probability = prey_mating_probability

    def mating(self, rewards, dones):
        """Optimized mating function using mating_scope without unnecessary nesting."""
        random.shuffle(self.agents)
        available_positions = {(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1]) if
                               self.grid[x, y] == 0}
        paired_agents = set()
        frozen_agents = set()
        agent_positions = {agent.get_position(): agent for agent in self.agents}

        for agent in self.agents:
            if agent in paired_agents or not available_positions:
                continue

            ax, ay = agent.get_position()
            mating_range = range(-self.mating_scope, self.mating_scope + 1)
            possible_mates = {(ax + dx, ay + dy) for dx in mating_range for dy in mating_range if (dx, dy) != (0, 0)}
            possible_mates = {(x % self.grid_size[0], y % self.grid_size[1]) for x, y in possible_mates}
            mates = [agent_positions[pos] for pos in possible_mates if
                     pos in agent_positions and agent_positions[pos].role == agent.role and agent_positions[
                         pos] not in paired_agents]

            if mates:
                closest_mate = min(mates, key=lambda m: abs(m.get_position()[0] - ax) + abs(m.get_position()[1] - ay))
                mating_prob = self.predator_mating_probability if agent.role == 'predator' else self.prey_mating_probability
                if random.random() < mating_prob:
                    rewards[agent.id] += self.mating_reward
                    rewards[closest_mate.id] += self.mating_reward
                    paired_agents.update([agent, closest_mate])
                    frozen_agents.update([agent, closest_mate])

                    if available_positions:
                        new_x, new_y = available_positions.pop()
                        new_agent_id = f"{'pr' if agent.role == 'predator' else 'py'}_{len([a for a in self.agents if a.role == agent.role])}"
                        new_agent = Agent(new_agent_id, agent.role, (new_x, new_y))
                        self.agents.append(new_agent)
                        self.grid[new_x, new_y] = new_agent

                        rewards[new_agent_id] = 0
                        dones[new_agent_id] = False

        return frozen_agents

    def ensure_population(self):
        """Ensure at least one predator and one prey are added each timestep."""
        available_positions = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1]) if
                               self.grid[x, y] == 0]

        if not available_positions:
            return

        random.shuffle(available_positions)  # Ensures random placement of new agents

        # Add at least one prey and one predator
        for role in ['predator', 'prey']:
            if len([a for a in self.agents if a.role == role]) < (
            self.max_num_predators if role == 'predator' else self.max_num_preys):
                if available_positions:  # Make sure there is available space
                    new_x, new_y = available_positions.pop()
                    new_agent_id = f"{'pr' if role == 'predator' else 'py'}_{len([a for a in self.agents if a.role == role])}"
                    new_agent = Agent(new_agent_id, role, (new_x, new_y))
                    self.agents.append(new_agent)
                    self.grid[new_x, new_y] = new_agent

    def predator_hunger(self, dones):
        """Decrease predator health and remove dead predators."""
        for predator in list(self.agents):
            if "predator" in predator.role:
                predator.add_health(-0.01)
                if predator.health <= 0:
                    px, py = predator.get_position()
                    self.agents.remove(predator)
                    self.grid[px, py] = 0
                    dones[predator.id] = True

        return dones
    def step(self, actions):
        """Takes a step in the environment based on the actions and environment rules."""
        rewards = {agent.id: 0 for agent in self.agents}
        dones = {agent.id: False for agent in self.agents}

        self.agents_move(actions)

        rewards, dones = self.hunting(rewards, dones)

        dones = self.predator_hunger(dones)

        self.mating(rewards, dones)

        self.ensure_population()

        # Update observations
        observations = {agent.id: self.get_observation(agent) for agent in self.agents}

        return observations, rewards, dones


    def generate_new_agents(self, p_predator=0.003, p_prey=0.006):
        pass
