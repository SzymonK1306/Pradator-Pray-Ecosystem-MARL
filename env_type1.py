import math

from pettingzoo.utils.env import ParallelEnv
import numpy as np
import random

from agent import Agent

class PredatorPreyEnv(ParallelEnv):
    def __init__(self,
                 grid_size=(600, 600),
                 num_predators=1000,
                 num_prey=1000,
                 num_walls=1000,
                 predator_scope=5,
                 health_gained=0.3,
                 heath_penalty=0.01):
        """
          - Initializes the environment.
          - grid_size: Tuple[int, int] - dimensions of the grid.
          - num_predators: int - number of predator agents.
          - num_prey: int - number of prey agents.
          - num_walls: int - number of wall elements.
          - predator_scope: int - range of predator, where preys are killed
          - health_gained: float - value of health restored with killing a prey
          - health_penalty: float - value of health losing by predator in each step
        """
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_walls = num_walls
        self.predator_scope = predator_scope
        self.health_gained = health_gained
        self.health_penalty = heath_penalty

        self.max_num_predators = 10000
        self.max_num_preys = 10000

        self.agents = []
        self.walls_positions = []

        # Initialize the grid
        self.grid = np.zeros(self.grid_size, dtype=object)

        # self.reset()

    def reset(self):
        """Resets the environment."""
        self.grid.fill(0)
        self.walls_positions.clear()

        # Place walls
        for _ in range(self.num_walls):
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = -1  # Wall
                    self.walls_positions.append((x, y))
                    break

        # Create and place predators
        for i in range(self.num_predators):
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    predator = Agent(f"pr_{i}", "predator", (x, y))
                    self.agents.append(predator)
                    self.grid[x, y] = predator  # Predator
                    break

        # Create and place preys
        for i in range(self.num_prey):
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    prey = Agent(f"py_{i}", "prey", (x, y))
                    self.agents.append(prey)
                    self.grid[x, y] = prey  # Prey
                    break

        return {agent.id: self.get_observation(agent) for agent in self.agents}

    def agents_move(self, actions):
        """Make a move of each agent"""
        new_positions = {}

        for agent in self.agents:
            x, y = agent.get_position()
            new_x, new_y = x, y

            # random actions for now
            action = actions[agent.id]

            if action == 1:  # up
                new_x = (x - 1) % self.grid_size[0]
            elif action == 2:  # down
                new_x = (x + 1) % self.grid_size[0]
            elif action == 3:  # left
                new_y = (y - 1) % self.grid_size[1]
            elif action == 4:  # right
                new_y = (y + 1) % self.grid_size[1]

            if self.grid[new_x, new_y] == 0:  # Move if the cell is empty
                new_positions[agent.id] = (new_x, new_y)
            else:  # Stay in place if the cell is occupied
                new_positions[agent.id] = (x, y)

        # Update grid and agent positions
        self.grid.fill(0)
        for wall in self.walls_positions:
            self.grid[wall[0], wall[1]] = -1

        for agent in self.agents:
            x, y = new_positions[agent.id]
            self.grid[x, y] = agent
            agent.set_position((x, y))

    def get_observation(self, agent):
        """
        Returns a local observation (7 channels):
          0: wall layer,
          1: predator layer,
          2: prey layer,
          3: health level,
        Observation size: (5*predator_scope+1, 5*predator_scope+1, 4)
        """
        ax, ay = agent.get_position()
        size = self.predator_scope * 8 + 1

        wall_layer = np.zeros((size, size), dtype=int)
        predator_layer = np.zeros((size, size), dtype=int)
        prey_layer = np.zeros((size, size), dtype=int)
        health_layer = np.zeros((size, size), dtype=float)

        for dx in range(-5 * self.predator_scope, 5 * self.predator_scope + 1):
            for dy in range(-5 * self.predator_scope, 5 * self.predator_scope + 1):
                nx, ny = (ax + dx) % self.grid_size[0], (ay + dy) % self.grid_size[1]
                local_x, local_y = dx + self.predator_scope, dy + self.predator_scope

                if self.grid[nx, ny] == -1:
                    wall_layer[local_x, local_y] = 1
                elif type(self.grid[nx, ny]) == Agent and self.grid[nx, ny].role == 'predator':
                    predator_layer[local_x, local_y] = 1
                    health_layer[local_x, local_y] = self.grid[nx, ny].health
                elif type(self.grid[nx, ny]) == Agent and self.grid[nx, ny].role == 'prey':
                    prey_layer[local_x, local_y] = 1
                    health_layer[local_x, local_y] = self.grid[nx, ny].health


        return np.stack([wall_layer, predator_layer, prey_layer, health_layer], axis=0)
    def hunting(self, rewards, dones):
        """Handle predator prey interaction - hunting"""
        for predator in [a for a in self.agents if "predator" in a.role]:
            px, py = predator.get_position()
            prey_in_scope = []

            for dx in range(-self.predator_scope, self.predator_scope + 1):
                for dy in range(-self.predator_scope, self.predator_scope + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = (px + dx) % self.grid_size[0], (py + dy) % self.grid_size[1]
                    if type(self.grid[nx, ny]) == Agent and self.grid[nx, ny].role == 'prey':
                        distance = abs(dx) + abs(dy)  # Manhattan
                        prey_in_scope.append((distance, (nx, ny)))

            if prey_in_scope:
                # Kill the nearest prey
                prey_in_scope.sort()
                target_prey_pos = prey_in_scope[0][1]
                for prey in self.agents:
                    pos = prey.get_position()
                    if pos == target_prey_pos:
                        self.agents.remove(prey)
                        self.grid[target_prey_pos[0], target_prey_pos[1]] = 0
                        rewards[predator.id] += 1  # Reward for eating prey
                        rewards[prey.id] += -1
                        predator.add_health(self.health_gained)  # Add constant value
                        dones[prey.id] = True
                        break

        return rewards, dones

    def predator_hunger(self, dones):
        """Decrease predator health and remove dead predators"""
        for predator in [a for a in self.agents if "predator" in a.role]:
            predator.add_health(-self.health_penalty)
            if predator.health <= 0:
                px, py = predator.get_position()
                self.agents.remove(predator)
                self.grid[px, py] = 0
                dones[predator.id] = True
        return dones

    def generate_new_agents(self, p_predator=0.003, p_prey=0.006):
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
        num_predators = len([a for a in self.agents if "predator" in a.role])
        num_preys = len([a for a in self.agents if "prey" in a.role])

        new_preys = 0
        new_predators = 0
        if num_predators < self.max_num_predators:
            new_predators = max(1, math.ceil(num_predators * p_predator))
        if num_preys < self.max_num_preys:
            new_preys = max(1, math.ceil(num_preys * p_prey))

        # Add new predators
        for _ in range(new_predators):
            predator_id = f"pr_{len([a for a in self.agents if 'predator' in a.role])}"
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:  # Empty cell
                    created_agent = Agent(predator_id, 'predator', (x, y))
                    self.grid[x, y] = created_agent  # Predator
                    self.agents.append(created_agent)
                    break

        # Add new preys
        for _ in range(new_preys):
            prey_id = f"py_{len([a for a in self.agents if 'prey' in a.role])}"
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:  # Empty cell
                    created_agent = Agent(prey_id, 'prey', (x, y))
                    self.grid[x, y] = created_agent  # Prey
                    self.agents.append(created_agent)
                    break

    def step(self, actions):
        """
        Performs one simulation step:
          1. Movement of agents
          2. Predator attack (hunting)
          3. Health decrease (hunger) for predators
          4. Generation of new agents
          5. Observation update
        """
        rewards = {agent.id: 0 for agent in self.agents}
        dones = {agent.id: False for agent in self.agents}

        self.agents_move(actions)

        rewards, dones = self.hunting(rewards, dones)

        dones = self.predator_hunger(dones)

        self.generate_new_agents(0.003, 0.006)
        # Update observations
        observations = {agent.id: self.get_observation(agent) for agent in self.agents}

        return observations, rewards, dones

    def render(self):
        """Renders the environment in the console.:
          - '#' representing wall,
          - 'X' representing predator,
          - 'O' representing prey.
        """
        render_grid = np.full(self.grid.shape, '.')

        render_grid[self.grid == -1] = '#'  # Wall
        render_grid[self.grid == 1] = 'O'  # Prey
        render_grid[self.grid == 2] = 'X'  # Predator

        print("\n".join("".join(row) for row in render_grid))
        print()