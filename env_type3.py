import random
import numpy as np
import math
from pettingzoo.utils.env import ParallelEnv
from agent_type3 import AgentType3

# ------------------------------
# Environment 3 – Unique genetic features for each agent
# ------------------------------
class PredatorPreyEnvType3(ParallelEnv):
    def __init__(self,
                 grid_size=(600, 600),
                 num_predators=20,
                 num_prey=30,
                 num_walls=500,
                 predator_scope=5,
                 health_gained=0.3,
                 health_penalty=0.01,
                 p_predator=0.003,
                 p_prey=0.006,
                 mating_scope=None,
                 mutation_chance=0.1,
                 mutation_std=1.0):
        """
          - Initializes the environment.
          - grid_size: Tuple[int, int] - dimensions of the grid.
          - num_predators: int - number of predator agents.
          - num_prey: int - number of prey agents.
          - num_walls: int - number of wall elements.
          - predator_scope: int - range of predator, where preys are killed
          - health_gained: float - value of health restored with killing a prey
          - health_penalty: float - value of health losing by predator in each step
          - p_predator: coefficient of generating new predators
          - p_prey: coefficient of generating new prey
          - mating_scope: reproduction range (defaults to the same as predator_scope if not provided)
          - mutation_chance: probability of mutation during recombination
          - mutation_std: standard deviation of the normal distribution used for mutation
        """
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_walls = num_walls
        self.predator_scope = predator_scope
        self.health_gained = health_gained
        self.health_penalty = health_penalty

        self.max_num_predators = 10000
        self.max_num_preys = 10000

        self.p_predator = p_predator
        self.p_prey = p_prey

        self.mutation_chance = mutation_chance
        self.mutation_std = mutation_std

        self.agents = []
        self.walls_positions = []
        self.grid = np.zeros(self.grid_size, dtype=object)

    def reset(self):
        """Resets the environment."""
        self.grid.fill(0)
        self.walls_positions.clear()
        self.agents = []

        # Create and place predators
        for _ in range(self.num_walls):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = -1  # Wall
                    self.walls_positions.append((x, y))
                    break

        # Create and place predators
        for i in range(self.num_predators):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    predator = AgentType3(f"pr_{i}", "predator", (x, y))
                    self.agents.append(predator)
                    self.grid[x, y] = predator
                    break

        # Create and place preys
        for i in range(self.num_prey):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    prey = AgentType3(f"py_{i}", "prey", (x, y))
                    self.agents.append(prey)
                    self.grid[x, y] = prey
                    break

        return {agent.id: self.get_observation(agent) for agent in self.agents}

    def agents_move(self, actions):
        """Make a move of each agent"""
        new_positions = {}
        for agent in self.agents:
            x, y = agent.get_position()
            new_x, new_y = x, y

            action = actions.get(agent.id, agent.get_random_action())
            if action == 1:  # up
                new_x = (x - 1) % self.grid_size[0]
            elif action == 2:  # down
                new_x = (x + 1) % self.grid_size[0]
            elif action == 3:  # left
                new_y = (y - 1) % self.grid_size[1]
            elif action == 4:  # right
                new_y = (y + 1) % self.grid_size[1]

            # Update empty cell
            if self.grid[new_x, new_y] == 0:
                new_positions[agent.id] = (new_x, new_y)
            else:
                new_positions[agent.id] = (x, y)

        # Update grid and agent positions
        self.grid.fill(0)
        for wx, wy in self.walls_positions:
            self.grid[wx, wy] = -1

        for agent in self.agents:
            pos = new_positions[agent.id]
            self.grid[pos[0], pos[1]] = agent
            agent.set_position(pos)

    def get_observation(self, agent):
        """
        Returns a local observation (7 channels):
          0: wall layer,
          1: predator layer,
          2: prey layer,
          3: health level,
          4: attack value (for predators; otherwise 0),
          5: resilience value (for prey; otherwise 0),
          6: speed value.
        Observation size: (5*predator_scope+1, 5*predator_scope+1, 7)
        """
        ax, ay = agent.get_position()
        size = self.predator_scope * 2 + 1

        wall_layer = np.zeros((size, size), dtype=int)
        predator_layer = np.zeros((size, size), dtype=int)
        prey_layer = np.zeros((size, size), dtype=int)
        health_layer = np.zeros((size, size), dtype=float)
        attack_layer = np.zeros((size, size), dtype=float)
        resilience_layer = np.zeros((size, size), dtype=float)
        speed_layer = np.zeros((size, size), dtype=float)

        for dx in range(-5 * self.predator_scope, 5 * self.predator_scope + 1):
            for dy in range(-5 * self.predator_scope, 5 * self.predator_scope + 1):
                nx = (ax + dx) % self.grid_size[0]
                ny = (ay + dy) % self.grid_size[1]
                local_x, local_y = dx + self.predator_scope, dy + self.predator_scope

                if self.grid[nx, ny] == -1:
                    wall_layer[local_x, local_y] = 1
                elif isinstance(self.grid[nx, ny], AgentType3):
                    other = self.grid[nx, ny]
                    if other.role == 'predator':
                        predator_layer[local_x, local_y] = 1
                        health_layer[local_x, local_y] = other.health
                        attack_layer[local_x, local_y] = other.attack
                        speed_layer[local_x, local_y] = other.speed
                    elif other.role == 'prey':
                        prey_layer[local_x, local_y] = 1
                        health_layer[local_x, local_y] = other.health
                        resilience_layer[local_x, local_y] = other.resilience
                        speed_layer[local_x, local_y] = other.speed

        observation = np.stack([wall_layer,
                                predator_layer,
                                prey_layer,
                                health_layer,
                                attack_layer,
                                resilience_layer,
                                speed_layer], axis=0)
        return observation

    def hunting(self, rewards, dones):
        """
        Predators search for the nearest prey (within predator_scope).
        All predators that attacked the same prey are assigned a share of the reward of 1,
        provided that the sum of their attack reduces the prey's resilience to 0 or below.
        """
        prey_attacks = {}
        for predator in [a for a in self.agents if a.role == 'predator']:
            px, py = predator.get_position()
            closest_prey = None
            min_distance = float('inf')
            for dx in range(-self.predator_scope, self.predator_scope + 1):
                for dy in range(-self.predator_scope, self.predator_scope + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (px + dx) % self.grid_size[0]
                    ny = (py + dy) % self.grid_size[1]
                    if isinstance(self.grid[nx, ny], AgentType3):
                        other = self.grid[nx, ny]
                        if other.role == 'prey':
                            distance = abs(dx) + abs(dy)
                            if distance < min_distance:
                                min_distance = distance
                                closest_prey = other
            if closest_prey is not None:
                prey_attacks.setdefault(closest_prey, []).append(predator)

        for prey, predators in prey_attacks.items():
            total_attack = sum(pred.attack for pred in predators)
            prey.resilience -= total_attack
            if prey.resilience <= 0:
                reward_share = 1.0 / len(predators)
                for pred in predators:
                    rewards[pred.id] += reward_share
                rewards[prey.id] += -1
                x, y = prey.get_position()
                if prey in self.agents:
                    self.agents.remove(prey)
                self.grid[x, y] = 0
                dones[prey.id] = True

        return rewards, dones

    def predator_hunger(self, dones):
        """Decrease predator health and remove dead predators"""
        for predator in list(a for a in self.agents if a.role == 'predator'):
            predator.add_health(-self.health_penalty)
            if predator.health <= 0:
                x, y = predator.get_position()
                self.agents.remove(predator)
                self.grid[x, y] = 0
                dones[predator.id] = True
        return dones

    def generate_new_agents(self):
        """
        Generates new agents based on the equation:
        N_new_agent = max(1, ceil(N_agent * p_agent))

        For predators, the combined parameters are speed and attack.
        For prey, the combined parameters are speed and resilience.
        The parameters of new agents result from the recombination
        (with potential mutation) of the parameters of two randomly
        selected parents.
        """
        # Calculate the number of new predators and prey
        predators = [a for a in self.agents if a.role == 'predator']
        num_predators = len(predators)
        new_predators = 0
        if num_predators < self.max_num_predators:
            new_predators = max(1, math.ceil(num_predators * self.p_predator))

        # Add new predators
        for _ in range(new_predators):
            if len(predators) >= 2:
                parent1, parent2 = random.sample(predators, 2)
                r = random.uniform(0, 1)
                new_speed = r * parent1.speed + (1 - r) * parent2.speed
                new_attack = r * parent1.attack + (1 - r) * parent2.attack
                # Mutation
                if random.random() < self.mutation_chance:
                    new_speed += np.random.normal(0, self.mutation_std)
                if random.random() < self.mutation_chance:
                    new_attack += np.random.normal(0, self.mutation_std)
            else:
                #
                # In cases where two parents are not available, agents are initialized randomly
                new_speed = random.uniform(0.5, 1.5)
                new_attack = random.uniform(0.5, 1.5)

            predator_id = f"pr_{len([a for a in self.agents if a.role == 'predator'])}"
            #
            # We search for a random, free position on the grid.
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    new_pred = AgentType3(predator_id, 'predator', (x, y))
                    new_pred.health = random.uniform(0.5, 1)
                    new_pred.speed = new_speed
                    new_pred.attack = new_attack
                    new_pred.resilience = 0  # For predators, resilience is not used.
                    self.agents.append(new_pred)
                    self.grid[x, y] = new_pred
                    break

        # --- Generating new preys ---
        preys = [a for a in self.agents if a.role == 'prey']
        num_preys = len(preys)
        new_preys = 0
        if num_preys < self.max_num_preys:
            new_preys = max(1, math.ceil(num_preys * self.p_prey))

        for _ in range(new_preys):
            if len(preys) >= 2:
                parent1, parent2 = random.sample(preys, 2)
                r = random.uniform(0, 1)
                new_speed = r * parent1.speed + (1 - r) * parent2.speed
                new_resilience = r * parent1.resilience + (1 - r) * parent2.resilience
                if random.random() < self.mutation_chance:
                    new_speed += np.random.normal(0, self.mutation_std)
                if random.random() < self.mutation_chance:
                    new_resilience += np.random.normal(0, self.mutation_std)
            else:
                new_speed = random.uniform(0.5, 1.5)
                new_resilience = random.uniform(0.5, 1.5)

            prey_id = f"py_{len([a for a in self.agents if a.role == 'prey'])}"
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    new_prey = AgentType3(prey_id, 'prey', (x, y))
                    new_prey.health = 1
                    new_prey.speed = new_speed
                    new_prey.resilience = new_resilience
                    new_prey.attack = 0  # For preys, attack is not used
                    self.agents.append(new_prey)
                    self.grid[x, y] = new_prey
                    break

    def step(self, actions):
        """
        Performs one simulation step:
          1. Movement of agents
          2. Predator attack (hunting)
          3. Health decrease (hunger) for predators
          4. Generation of new agents (only if there are pairs within the reproduction range)
          5. Observation update
        """
        rewards = {agent.id: 0 for agent in self.agents}
        dones = {agent.id: False for agent in self.agents}

        self.agents_move(actions)
        rewards, dones = self.hunting(rewards, dones)
        dones = self.predator_hunger(dones)
        self.generate_new_agents()
        observations = {agent.id: self.get_observation(agent) for agent in self.agents}
        return observations, rewards, dones

    def render(self):
        """
        Renders the environment in the console.:
          - '#' representing wall,
          - 'X' representing predator,
          - 'O' representing prey.
        """
        render_grid = np.full(self.grid.shape, '.')
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] == -1:
                    render_grid[i, j] = '#'
                elif isinstance(self.grid[i, j], AgentType3):
                    if self.grid[i, j].role == 'predator':
                        render_grid[i, j] = 'X'
                    elif self.grid[i, j].role == 'prey':
                        render_grid[i, j] = 'O'
        for row in render_grid:
            print("".join(row))
        print()
