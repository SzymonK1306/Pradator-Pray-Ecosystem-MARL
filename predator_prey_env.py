
from pettingzoo.utils.env import ParallelEnv
import numpy as np
import random
from agent import Agent
import math

class PredatorPreyEnv(ParallelEnv):
    def __init__(self, grid_size=(15, 15), num_predators=2, num_prey=3, num_walls=5, predator_scope=2, health_gained=0.3):
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

        self.max_num_predators = 10000
        self.max_num_preys = 10000

        self.agents = []
        # self.agent_positions = {agent: None for agent in self.agents}
        # self.agent_health = {agent: 1 for agent in self.agents}
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

        # Create and place prey
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

    # TODO try to optimise with object pointers
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
                        # print(f'{prey.id} killed')
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
                # print(f'{predator.id} killed')
        return dones

    def generate_new_agents(self, p_predator=0.003, p_prey=0.006):
        """
        Generates new predators and prey based on the provided formula.
        p_predator: float - probability factor for generating new predators.
        p_prey: float - probability factor for generating new prey.
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
        """Takes a step in the environment based on the actions and environment rules."""
        rewards = {agent.id: 0 for agent in self.agents}
        dones = {agent.id: False for agent in self.agents}

        self.agents_move(actions)

        rewards, dones = self.hunting(rewards, dones)

        dones = self.predator_hunger(dones)

        self.generate_new_agents(0.003, 0.006)
        # Update observations
        observations = {agent.id: self.get_observation(agent) for agent in self.agents}

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
                elif type(self.grid[nx, ny]) == Agent and self.grid[nx, ny].role == 'predator':
                    predator_layer[local_x, local_y] = 1
                    health_layer[local_x, local_y] = self.grid[nx, ny].health
                elif type(self.grid[nx, ny]) == Agent and self.grid[nx, ny].role == 'prey':
                    prey_layer[local_x, local_y] = 1
                    health_layer[local_x, local_y] = self.grid[nx, ny].health


        return np.stack([wall_layer, predator_layer, prey_layer, health_layer], axis=0)

    def render(self):
        """Renders the environment in the console."""
        render_grid = np.full(self.grid.shape, '.')

        render_grid[self.grid == -1] = '#'  # Wall
        render_grid[self.grid == 1] = 'O'  # Prey
        render_grid[self.grid == 2] = 'X'  # Predator

        print("\n".join("".join(row) for row in render_grid))
        print()


class PredatorPreyEnvType2(PredatorPreyEnv):
    def __init__(self, grid_size=(600, 600), num_predators=2000, num_prey=1000, num_walls=10, predator_scope=5,
                 health_gained=1.0, mating_scope=15, mating_reward=4, predator_mating_probability=0.003,
                 prey_mating_probability=0.006):
        """
        Initializes the extended environment for Type 2 simulation with mating behavior and environment settings.
        """
        super().__init__(grid_size, num_predators, num_prey, num_walls, predator_scope, health_gained)
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

                        # 🛠 **Naprawa KeyError - dodanie nowego agenta do `rewards` oraz `dones`** 🛠
                        rewards[new_agent_id] = 0
                        dones[new_agent_id] = False

                        #print(f"{agent.role} mated and created {new_agent_id} at ({new_x}, {new_y})")

        return frozen_agents

    def ensure_population(self):
        """Ensure at least one predator and one prey are added each timestep."""
        available_positions = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1]) if
                               self.grid[x, y] == 0]

        if not available_positions:
            return  # Brak wolnych miejsc w siatce

        random.shuffle(available_positions)  # Zapewnia losowe rozmieszczenie nowych agentów

        # Dodaj co najmniej jednego preya i jednego predatora
        for role in ['predator', 'prey']:
            if len([a for a in self.agents if a.role == role]) < (
            self.max_num_predators if role == 'predator' else self.max_num_preys):
                if available_positions:  # Upewnij się, że mamy dostępne miejsce
                    new_x, new_y = available_positions.pop()
                    new_agent_id = f"{'pr' if role == 'predator' else 'py'}_{len([a for a in self.agents if a.role == role])}"
                    new_agent = Agent(new_agent_id, role, (new_x, new_y))
                    self.agents.append(new_agent)
                    self.grid[new_x, new_y] = new_agent
                    #print("added")
                    #print(role)
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
        # Update observations
        observations = {agent.id: self.get_observation(agent) for agent in self.agents}

        return observations, rewards, dones


    def generate_new_agents(self, p_predator=0.003, p_prey=0.006):
        pass
