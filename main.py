from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers
import numpy as np
import random
from agent import Agent

import json

class PredatorPreyEnv(ParallelEnv):
    def __init__(self, grid_size=(10, 10), num_predators=2, num_prey=3, num_walls=5, predator_scope=2, health_gained=0.3):
        """
        Initializes the environment.
        grid_size: Tuple[int, int] - dimensions of the grid.
        num_predators: int - number of predator agents.
        num_prey: int - number of prey agents.
        num_walls: int - number of wall elements.
        predator_scope: int - range of predator, where preys are killed
        health_gained: float - value of health restored with killing a prey
        """
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_walls = num_walls
        self.predator_scope = predator_scope
        self.health_gained = health_gained

        self.agents = []
        # self.agent_positions = {agent: None for agent in self.agents}
        # self.agent_health = {agent: 1 for agent in self.agents}
        self.walls_positions = []

        # Initialize the grid
        self.grid = np.zeros(self.grid_size, dtype=int)

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
                    self.grid[x, y] = 2  # Predator
                    break

        # Create and place prey
        for i in range(self.num_prey):
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    prey = Agent(f"py_{i}", "prey", (x, y))
                    self.agents.append(prey)
                    self.grid[x, y] = 1  # Prey
                    break

        return {agent: self.get_observation(agent) for agent in self.agents}

    def agents_move(self):
        """Make a move of each agent"""
        new_positions = {}

        for agent in self.agents:
            x, y = agent.get_position()
            new_x, new_y = x, y

            # random actions for now
            action = agent.get_random_action()

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
            if "predator" in agent.role:
                self.grid[x, y] = 2
            else:
                self.grid[x, y] = 1
            agent.set_position((x, y))

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
                    if self.grid[nx, ny] == 1:
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
                        print(f'{prey.id} killed')
                        break

        return rewards, dones

    def predator_hunger(self, dones):
        """Decrease predator health and remove dead predators"""
        for predator in [a for a in self.agents if "predator" in a.role]:
            predator.add_health(-0.01)
            if predator.health <= 0:
                px, py = predator.get_position()
                self.agents.remove(predator)
                self.grid[px, py] = 0
                dones[predator.id] = True
                print(f'{predator.id} killed')
                return dones

    def generate_new_agents(self, p_predator=0.1, p_prey=0.1):
        """
        Generates new predators and prey based on the provided formula.
        p_predator: float - probability factor for generating new predators.
        p_prey: float - probability factor for generating new prey.
        """
        # Calculate the number of new predators and prey
        num_predators = len([a for a in self.agents if "predator" in a])
        num_prey = len([a for a in self.agents if "prey" in a])

        new_predators = min(1, int(num_predators * p_predator))
        new_prey = min(1, int(num_prey * p_prey))

        # Add new predators
        for _ in range(new_predators):
            predator_id = f"pr_{len([a for a in self.agents if 'predator' in a.role])}"
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:  # Empty cell
                    self.grid[x, y] = 2  # Predator
                    self.agents.append(Agent(predator_id, 'predator', (x, y)))
                    break

        # Add new preys
        for _ in range(new_prey):
            prey_id = f"py_{len([a for a in self.agents if 'prey' in a.role])}"
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:  # Empty cell
                    self.grid[x, y] = 1  # Prey
                    self.agents.append(Agent(prey_id, 'prey', (x, y)))
                    break

    def step(self):
        """Takes a step in the environment based on the actions and environment rules."""
        rewards = {agent.id: 0 for agent in self.agents}
        dones = {agent.id: False for agent in self.agents}

        self.agents_move()

        rewards, dones = self.hunting(rewards, dones)

        dones = self.predator_hunger(dones)

        # Update observations
        observations = {agent.id: self.get_observation(agent) for agent in self.agents}

        # self.generate_new_agents()  # TODO Check formula in publication, seems to be wrong

        return observations, rewards, dones

    def get_observation(self, agent):
        """Returns a 4-channel local grid observation for the given agent."""
        ax, ay = agent.get_position()
        size = self.predator_scope * 2 + 1

        wall_layer = np.zeros((size, size), dtype=int)
        predator_layer = np.zeros((size, size), dtype=int)
        prey_layer = np.zeros((size, size), dtype=int)
        health_layer = np.zeros((size, size), dtype=float)

        for dx in range(-self.predator_scope, self.predator_scope + 1):
            for dy in range(-self.predator_scope, self.predator_scope + 1):
                nx, ny = (ax + dx) % self.grid_size[0], (ay + dy) % self.grid_size[1]
                local_x, local_y = dx + self.predator_scope, dy + self.predator_scope

                if self.grid[nx, ny] == -1:
                    wall_layer[local_x, local_y] = 1
                elif self.grid[nx, ny] == 2:
                    predator_layer[local_x, local_y] = 1
                elif self.grid[nx, ny] == 1:
                    prey_layer[local_x, local_y] = 1

                for a in self.agents:
                    px, py = a.get_position()
                    if (px, py) == (nx, ny):
                        health_layer[local_x, local_y] = a.health

        return np.stack([wall_layer, predator_layer, prey_layer, health_layer], axis=0)

    def render(self):
        """Renders the environment in the console."""
        render_grid = np.full(self.grid.shape, '.')

        render_grid[self.grid == -1] = '#'  # Wall
        render_grid[self.grid == 1] = 'O'  # Prey
        render_grid[self.grid == 2] = 'X'  # Predator

        print("\n".join("".join(row) for row in render_grid))
        print()

# Wrapping the environment - Can be added in the future

def env_creator():
    env = PredatorPreyEnv()
    return env

# Example usage
if __name__ == "__main__":
    env = env_creator()
    obs = env.reset()
    env.render()

    # save experiences of agents
    experiences = {
    "observations": {},
    "actions": {},
    "rewards": {},
    "dones": {}
    }

    obs = {agent.id: env.get_observation(agent) for agent in env.agents}

    for i in range(20):
        actions = {agent: random.randint(0, 4) for agent in env.agents}
        new_obs, rewards, dones = env.step()

        experiences["observations"].update(obs)
        experiences["actions"].update(actions)
        experiences["rewards"].update(rewards)
        experiences["dones"].update(dones)

        obs = new_obs
        env.render()

# TODO Implement learning algorithm
# TODO Implement exploration algorithm

