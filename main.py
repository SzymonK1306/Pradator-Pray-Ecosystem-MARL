from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers
import numpy as np
import random

class PredatorPreyEnv(ParallelEnv):
    def __init__(self, grid_size=(10, 10), num_predators=2, num_prey=3, num_walls=5, predator_scope=2):
        """
        Initializes the environment.
        grid_size: Tuple[int, int] - dimensions of the grid.
        num_predators: int - number of predator agents.
        num_prey: int - number of prey agents.
        num_walls: int - number of wall elements.
        """
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_walls = num_walls
        self.predator_scope = predator_scope

        self.agents = [f"predator_{i}" for i in range(num_predators)] + [f"prey_{i}" for i in range(num_prey)]
        self.agent_positions = {agent: None for agent in self.agents}
        self.agent_health = {agent: 1 for agent in self.agents}
        self.walls_positions = []

        # Initialize the grid
        self.grid = np.zeros(self.grid_size, dtype=int)

        self.action_space = {agent: 5 for agent in self.agents}  # Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.observation_space = {agent: (3, 3) for agent in self.agents}  # 3x3 observation square

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

        # Place predators and prey
        for agent in self.agents:
            while True:
                x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    if "predator" in agent:
                        self.grid[x, y] = 2  # Predator
                    else:
                        self.grid[x, y] = 1  # Prey
                    self.agent_positions[agent] = (x, y)
                    break

        return {agent: self.get_observation(agent) for agent in self.agents}

    def step(self, actions):
        """Takes a step in the environment based on the actions."""
        rewards = {agent: 0 for agent in self.agents}

        new_positions = {}

        for agent, action in actions.items():
            x, y = self.agent_positions[agent]
            new_x, new_y = x, y

            if action == 1:  # up
                new_x = (x - 1) % self.grid_size[0]
            elif action == 2:  # down
                new_x = (x + 1) % self.grid_size[0]
            elif action == 3:  # left
                new_y = (y - 1) % self.grid_size[1]
            elif action == 4:  # right
                new_y = (y + 1) % self.grid_size[1]

            if self.grid[new_x, new_y] == 0:  # Move if the cell is empty
                new_positions[agent] = (new_x, new_y)
            else:  # Stay in place if the cell is occupied
                new_positions[agent] = (x, y)

        # Update grid and agent positions
        self.grid.fill(0)
        for wall in self.walls_positions:
            self.grid[wall[0], wall[1]] = -1

        for agent, (x, y) in new_positions.items():
            if "predator" in agent:
                self.grid[x, y] = 2
            else:
                self.grid[x, y] = 1
            self.agent_positions[agent] = (x, y)

        # Handle predator-prey interactions
        for predator in [a for a in self.agents if "predator" in a]:
            px, py = self.agent_positions[predator]
            prey_in_scope = []

            for dx in range(-self.predator_scope, self.predator_scope + 1):
                for dy in range(-self.predator_scope, self.predator_scope + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = (px + dx) % self.grid_size[0], (py + dy) % self.grid_size[1]
                    if self.grid[nx, ny] == 1:
                        prey_in_scope.append((nx, ny))

            if prey_in_scope:
                # Kill the nearest prey (or the first one found in scope for simplicity)
                target_prey_pos = prey_in_scope[0]
                for prey, pos in self.agent_positions.items():
                    if pos == target_prey_pos:
                        del self.agent_positions[prey]
                        self.agents.remove(prey)
                        self.grid[target_prey_pos[0], target_prey_pos[1]] = 0
                        rewards[predator] += 1  # Reward for eating prey
                        print(f'{prey} killed')
                        break

        # Decrease predator health and remove dead predators
        for predator in [a for a in self.agents if "predator" in a]:
            self.agent_health[predator] -= 0.01
            if self.agent_health[predator] <= 0:
                px, py = self.agent_positions[predator]
                del self.agent_positions[predator]
                self.agents.remove(predator)
                self.grid[px, py] = 0
                print(f'{predator} killed')


        # Update observations
        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: False for agent in self.agents}

        return observations, rewards, dones, {}

    def get_observation(self, agent):
        """Returns a 3x3 grid observation for the given agent."""
        x, y = self.agent_positions[agent]
        half_size = 1  # Half of 3x3 observation window
        obs = np.zeros((3, 3), dtype=int)

        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                nx, ny = (x + dx) % self.grid_size[0], (y + dy) % self.grid_size[1]
                obs[dx + half_size, dy + half_size] = self.grid[nx, ny]

        return obs

    def render(self):
        """Renders the environment in the console."""
        render_grid = np.full(self.grid.shape, '.')

        render_grid[self.grid == -1] = '#'  # Wall
        render_grid[self.grid == 1] = 'O'  # Prey
        render_grid[self.grid == 2] = 'X'  # Predator

        print("\n".join("".join(row) for row in render_grid))
        print()

# Wrapping the environment

def env_creator():
    env = PredatorPreyEnv()
    return env

# Example usage
if __name__ == "__main__":
    env = env_creator()
    obs = env.reset()
    env.render()

    for i in range(20):
        actions = {agent: random.randint(0, 4) for agent in env.agents}
        obs, rewards, dones, infos = env.step(actions)
        env.render()

# TODO Change observation function
# TODO Implement learning algorithm
# TODO Implement exploration algorithm

