import random
import numpy as np
import math
from pettingzoo.utils.env import ParallelEnv


# ------------------------------
# Klasa Agent – podstawowa wersja
# ------------------------------
class Agent():
    def __init__(self, id, role, position):
        self.id = id  # Unikalne ID agenta
        self.role = role
        self.position = position

        if role == 'predator':
            self.health = random.uniform(0.5, 1)
            self.speed = random.uniform(0.5, 1.5)
            self.attack = random.uniform(0.5, 1.5)
            self.resilience = 0  # Dla drapieżników nie jest używane
        else:
            self.health = 1
            self.speed = random.uniform(0.5, 1.5)
            self.resilience = random.uniform(0.5, 1.5)
            self.attack = 0  # Ofiary nie posiadają atrybutu attack

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def get_random_action(self):
        # Akcje: 1 = up, 2 = down, 3 = left, 4 = right
        return random.choice([1, 2, 3, 4])

    def add_health(self, health_gained):
        self.health += health_gained


# ------------------------------
# Środowisko typu 3 – generowanie nowych osobników tylko po znalezieniu pary w zasięgu reprodukcji
# ------------------------------
class PredatorPreyEnvType3(ParallelEnv):
    def __init__(self,
                 grid_size=(600, 600),
                 num_predators=500,
                 num_prey=100,
                 num_walls=1000,
                 predator_scope=5,
                 health_gained=0.3,
                 mating_scope=10,
                 mutation_chance=0.1,
                 mutation_std=1.0):
        """
        Inicjalizacja środowiska typu 3.

        Parametry:
          - grid_size: rozmiar siatki (wysokość, szerokość)
          - num_predators: początkowa liczba drapieżników
          - num_prey: początkowa liczba ofiar
          - num_walls: liczba ścian
          - predator_scope: zasięg widzenia/poszukiwania (obserwacja: (2*predator_scope+1)^2)
          - health_gained: przyrost zdrowia drapieżnika po skutecznym ataku
          - mating_scope: maksymalna odległość, przy której dwa agenty mogą się spotkać i zreprodukować;
                          domyślnie przyjmujemy tę samą wartość co predator_scope (można podać oddzielnie)
          - mutation_chance: prawdopodobieństwo wystąpienia mutacji podczas recombinacji
          - mutation_std: odchylenie standardowe rozkładu normalnego używanego przy mutacji
        """
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_walls = num_walls
        self.predator_scope = predator_scope
        self.health_gained = health_gained

        # Ustalanie maksymalnej liczby agentów (przydatne przy globalnych warunkach, aczkolwiek w naszym przypadku
        # generowanie nowych osobników zależy od spotkania par)
        self.max_num_predators = 10000
        self.max_num_preys = 10000

        # Zakres reprodukcyjny – jeśli nie podano, przyjmujemy wartość predator_scope
        self.mating_scope = mating_scope if mating_scope is not None else predator_scope

        self.mutation_chance = mutation_chance
        self.mutation_std = mutation_std

        self.agents = []
        self.walls_positions = []
        self.grid = np.zeros(self.grid_size, dtype=object)

    def reset(self):
        """Resetuje środowisko oraz inicjalizuje agentów."""
        self.grid.fill(0)
        self.walls_positions.clear()
        self.agents = []

        # Ustawianie ścian
        for _ in range(self.num_walls):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = -1  # Reprezentacja ściany
                    self.walls_positions.append((x, y))
                    break

        # Inicjalizacja drapieżników
        for i in range(self.num_predators):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    predator = Agent(f"pr_{i}", "predator", (x, y))
                    self.agents.append(predator)
                    self.grid[x, y] = predator
                    break

        # Inicjalizacja ofiar
        for i in range(self.num_prey):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    prey = Agent(f"py_{i}", "prey", (x, y))
                    self.agents.append(prey)
                    self.grid[x, y] = prey
                    break

        return {agent.id: self.get_observation(agent) for agent in self.agents}

    def agents_move(self, actions):
        """Przemieszcza agentów wg zadanych akcji."""
        new_positions = {}
        for agent in self.agents:
            x, y = agent.get_position()
            new_x, new_y = x, y

            # Pobieramy akcję – jeśli nie podano, wybieramy losowo
            action = actions.get(agent.id, agent.get_random_action())
            if action == 1:  # up
                new_x = (x - 1) % self.grid_size[0]
            elif action == 2:  # down
                new_x = (x + 1) % self.grid_size[0]
            elif action == 3:  # left
                new_y = (y - 1) % self.grid_size[1]
            elif action == 4:  # right
                new_y = (y + 1) % self.grid_size[1]

            # Jeżeli docelowa komórka jest pusta, przechodzimy do niej
            if self.grid[new_x, new_y] == 0:
                new_positions[agent.id] = (new_x, new_y)
            else:
                new_positions[agent.id] = (x, y)

        # Czyszczenie siatki – zachowujemy ściany
        self.grid.fill(0)
        for wx, wy in self.walls_positions:
            self.grid[wx, wy] = -1

        for agent in self.agents:
            pos = new_positions[agent.id]
            self.grid[pos[0], pos[1]] = agent
            agent.set_position(pos)

    def get_observation(self, agent):
        """
        Zwraca lokalną obserwację (7 kanałów):
          0: warstwa ścian,
          1: warstwa drapieżników,
          2: warstwa ofiar,
          3: poziom health,
          4: wartość attack (dla drapieżników; w innym przypadku 0),
          5: wartość resilience (dla ofiar; w innym przypadku 0),
          6: wartość speed.
        Wielkość obserwacji: (2*predator_scope+1, 2*predator_scope+1)
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

        for dx in range(-self.predator_scope, self.predator_scope + 1):
            for dy in range(-self.predator_scope, self.predator_scope + 1):
                nx = (ax + dx) % self.grid_size[0]
                ny = (ay + dy) % self.grid_size[1]
                local_x, local_y = dx + self.predator_scope, dy + self.predator_scope

                if self.grid[nx, ny] == -1:
                    wall_layer[local_x, local_y] = 1
                elif isinstance(self.grid[nx, ny], Agent):
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
        Drapieżniki szukają najbliższej ofiary (w obrębie predator_scope).
        Wszystkim drapieżnikom, które zaatakowały tę samą ofiarę, przypisywana jest część
        nagrody 1, jeśli suma ich attack powoduje zmniejszenie resilience ofiary do 0 lub mniej.
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
                    if isinstance(self.grid[nx, ny], Agent):
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
        """Zmniejsza zdrowie drapieżników (symulacja głodu) i usuwa te, które umierają."""
        for predator in list(a for a in self.agents if a.role == 'predator'):
            predator.add_health(-0.01)
            if predator.health <= 0:
                x, y = predator.get_position()
                self.agents.remove(predator)
                self.grid[x, y] = 0
                dones[predator.id] = True
        return dones

    def generate_new_agents(self):
        """
        Generuje nowych agentów poprzez recombinację parametrów dwóch agentów, ale tylko wtedy,
        gdy para agentów tego samego typu znajduje się w zasięgu reprodukcyjnym (mating scope).

        Dla drapieżników łączone są parametry: speed oraz attack.
        Dla ofiar łączone są parametry: speed oraz resilience.
        Nowy agent pojawia się tylko dla par, które spełniają warunek odległości mniejszej lub równej mating_scope.
        """

        # --- Dla drapieżników ---
        predators = [a for a in self.agents if a.role == 'predator']
        predator_pairs = []
        # Wyszukiwanie par drapieżników znajdujących się wystarczająco blisko siebie
        for i, agent1 in enumerate(predators):
            for agent2 in predators[i + 1:]:
                x1, y1 = agent1.get_position()
                x2, y2 = agent2.get_position()
                distance = abs(x1 - x2) + abs(y1 - y2)  # odległość Manhattan
                if distance <= self.mating_scope:
                    predator_pairs.append((agent1, agent2))
        # Rekombinacja parametrów dla każdej znalezionej pary
        for parent1, parent2 in predator_pairs:
            r = random.uniform(0, 1)
            new_speed = r * parent1.speed + (1 - r) * parent2.speed
            new_attack = r * parent1.attack + (1 - r) * parent2.attack
            # Mutacja
            if random.random() < self.mutation_chance:
                new_speed += np.random.normal(0, self.mutation_std)
            if random.random() < self.mutation_chance:
                new_attack += np.random.normal(0, self.mutation_std)
            # Nowy identyfikator nowego drapieżnika
            predator_id = f"pr_{len([a for a in self.agents if a.role == 'predator'])}"
            # Znalezienie losowej, wolnej pozycji na siatce
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    new_pred = Agent(predator_id, 'predator', (x, y))
                    new_pred.health = random.uniform(0.5, 1)
                    new_pred.speed = new_speed
                    new_pred.attack = new_attack
                    new_pred.resilience = 0
                    self.agents.append(new_pred)
                    self.grid[x, y] = new_pred
                    break

        # --- Dla ofiar ---
        preys = [a for a in self.agents if a.role == 'prey']
        prey_pairs = []
        for i, agent1 in enumerate(preys):
            for agent2 in preys[i + 1:]:
                x1, y1 = agent1.get_position()
                x2, y2 = agent2.get_position()
                distance = abs(x1 - x2) + abs(y1 - y2)
                if distance <= self.mating_scope:
                    prey_pairs.append((agent1, agent2))
        for parent1, parent2 in prey_pairs:
            r = random.uniform(0, 1)
            new_speed = r * parent1.speed + (1 - r) * parent2.speed
            new_resilience = r * parent1.resilience + (1 - r) * parent2.resilience
            if random.random() < self.mutation_chance:
                new_speed += np.random.normal(0, self.mutation_std)
            if random.random() < self.mutation_chance:
                new_resilience += np.random.normal(0, self.mutation_std)
            prey_id = f"py_{len([a for a in self.agents if a.role == 'prey'])}"
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    new_prey = Agent(prey_id, 'prey', (x, y))
                    new_prey.health = 1
                    new_prey.speed = new_speed
                    new_prey.resilience = new_resilience
                    new_prey.attack = 0
                    self.agents.append(new_prey)
                    self.grid[x, y] = new_prey
                    break

    def step(self, actions):
        """
        Wykonuje jeden krok symulacji:
          1. Ruch agentów
          2. Atak drapieżników (hunting)
          3. Spadek zdrowia (głód) drapieżników
          4. Generowanie nowych agentów (tylko jeśli istnieją pary w zasięgu reprodukcji)
          5. Aktualizacja obserwacji
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
        Renderuje środowisko w konsoli:
          - '#' reprezentuje ścianę,
          - 'X' reprezentuje drapieżnika,
          - 'O' reprezentuje ofiarę.
        """
        render_grid = np.full(self.grid.shape, '.')
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] == -1:
                    render_grid[i, j] = '#'
                elif isinstance(self.grid[i, j], Agent):
                    if self.grid[i, j].role == 'predator':
                        render_grid[i, j] = 'X'
                    elif self.grid[i, j].role == 'prey':
                        render_grid[i, j] = 'O'
        for row in render_grid:
            print("".join(row))
        print()
