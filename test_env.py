import unittest
import random
from unittest.mock import patch

from agent import Agent
from main import PredatorPreyEnv

class TestPredatorPreyEnv(unittest.TestCase):
    def setUp(self):
        self.env = PredatorPreyEnv(grid_size=(15, 15), num_predators=2, num_prey=3, 
                                   num_walls=5, predator_scope=2, health_gained=0.3)

    def test_initialization(self):
        self.env.reset()
        
        self.assertEqual(self.env.grid_size, (15, 15))
        self.assertEqual(self.env.num_predators, 2)
        self.assertEqual(self.env.num_prey, 3)
        self.assertEqual(self.env.num_walls, 5)
        self.assertEqual(self.env.predator_scope, 2)
        self.assertEqual(self.env.health_gained, 0.3)
        self.assertEqual(len(self.env.agents), 5)
        self.assertEqual(len(self.env.walls_positions), 5)

    def test_reset(self):
        initial_state = self.env.reset()
        
        self.assertEqual(len(self.env.agents), 5)
        self.assertEqual(len(self.env.walls_positions), 5)
        self.assertEqual(len(initial_state), 5)

    def test_agents_move(self):
        for _ in range(25):
            self.env.reset()

            actions = {agent.id: agent.get_random_action() for agent in self.env.agents}
            self.env.agents_move(actions)
            
            for agent in self.env.agents:
                self.assertIn(agent.get_position()[0], range(15))
                self.assertIn(agent.get_position()[1], range(15))

    def test_hunting(self):
        self.env.reset()
        
        predator = Agent("pr_0", "predator", (5, 5))
        prey = Agent("py_0", "prey", (6, 6))  # Prey is within predator's scope
        
        self.env.grid.fill(0)
        self.env.agents = [predator, prey]
        self.env.grid[5, 5] = predator
        self.env.grid[6, 6] = prey

        rewards = {agent.id: 0 for agent in self.env.agents}
        dones = {agent.id: False for agent in self.env.agents}

        rewards, dones = self.env.hunting(rewards, dones)

        self.assertEqual(rewards[predator.id], 1)  # Predator should get a reward
        self.assertEqual(rewards[prey.id], -1)     # Prey should get a penalty
        self.assertTrue(dones[prey.id])            # Prey should be marked as done (dead)
        self.assertEqual(len(self.env.agents), 1)  # Prey should be removed from the environment
        self.assertEqual(self.env.grid[6, 6], 0)   # Prey's position should be cleared

    def test_predator_hunger(self):
        self.env.reset()
        
        # Manually place a predator with low health
        predator = Agent("pr_0", "predator", (5, 5))
        predator.health = 0.01
        self.env.agents = [predator]
        self.env.grid[5, 5] = 2

        dones = {agent.id: False for agent in self.env.agents}
        dones = self.env.predator_hunger(dones)

        # Check if the predator died due to hunger
        self.assertTrue(dones[predator.id]) 
        self.assertEqual(len(self.env.agents), 0) 
        self.assertEqual(self.env.grid[5, 5], 0)  

        # Test health decrease over multiple steps
        predator = Agent("pr_0", "predator", (5, 5))
        predator.health = 1.0  # Reset health
        self.env.agents = [predator]
        self.env.grid[5, 5] = predator

        initial_health = predator.health
        for _ in range(15):
            dones = {agent.id: False for agent in self.env.agents}
            self.env.predator_hunger(dones)
            self.assertLess(predator.health, initial_health)  # Health should decrease
            initial_health = predator.health

    def test_generate_new_agents(self):
        self.env.reset()
        
        initial_predator_count = len([a for a in self.env.agents if "predator" in a.role])
        initial_prey_count = len([a for a in self.env.agents if "prey" in a.role])
        
        self.env.generate_new_agents(p_predator=0.1, p_prey=0.1)
        
        new_predator_count = len([a for a in self.env.agents if "predator" in a.role])
        new_prey_count = len([a for a in self.env.agents if "prey" in a.role])
        
        self.assertGreaterEqual(new_predator_count, initial_predator_count)
        self.assertGreaterEqual(new_prey_count, initial_prey_count)

    def test_step(self):
        observations = self.env.reset()
        
        initial_predator_count = len([a for a in self.env.agents if "predator" in a.role])
        initial_prey_count = len([a for a in self.env.agents if "prey" in a.role])
        initial_predator_health = [a.health for a in self.env.agents if "predator" in a.role][0]

        actions = {agent.id: agent.get_random_action() for agent in self.env.agents}
        new_observations, rewards, dones = self.env.step(actions)

        self.assertEqual(len(new_observations), len(self.env.agents))

        all_agent_ids = {agent.id for agent in self.env.agents}  
        all_reward_ids = set(rewards.keys()) 
        all_done_ids = set(dones.keys())  

        self.assertTrue(all_agent_ids.issubset(all_reward_ids))  # Living agents must have rewards
        self.assertTrue(all_agent_ids.issubset(all_done_ids))  # Living agents must have dones

        new_predator_health = [a.health for a in self.env.agents if "predator" in a.role][0]
        self.assertNotEqual(new_predator_health, initial_predator_health)  # Health should change

        # Check if the number of predators and prey has changed (due to hunting or starvation)
        new_predator_count = len([a for a in self.env.agents if "predator" in a.role])
        new_prey_count = len([a for a in self.env.agents if "prey" in a.role])
        self.assertLessEqual(new_predator_count, initial_predator_count)
        self.assertLessEqual(new_prey_count, initial_prey_count)

    def test_get_observation(self):
        self.env.reset()
        
        agent = self.env.agents[0]
        observation = self.env.get_observation(agent)
        
        self.assertEqual(observation.shape, (4, 5, 5))

    def test_render(self):
        self.env.reset()
        
        with patch('builtins.print') as mock_print:
            self.env.render()
            mock_print.assert_called()

    def test_generate_new_predators(self):
        self.env.reset()
        
        initial_predator_count = len([a for a in self.env.agents if "predator" in a.role])
        p_predator = 0.1
        
        for _ in range(15):
            self.env.generate_new_agents(p_predator=p_predator, p_prey=0.0)
            new_predator_count = len([a for a in self.env.agents if "predator" in a.role])
            expected_new_predators = min(1, int(initial_predator_count * p_predator))
            self.assertGreaterEqual(new_predator_count, initial_predator_count + expected_new_predators)
            initial_predator_count = new_predator_count
            
    def test_generate_new_prey(self):
        self.env.reset()
        
        initial_prey_count = len([a for a in self.env.agents if "prey" in a.role])
        p_prey = 0.1

        for _ in range(15):
            self.env.generate_new_agents(p_predator=0.0, p_prey=p_prey)
            new_prey_count = len([a for a in self.env.agents if "prey" in a.role])
            expected_new_prey = min(1, int(initial_prey_count * p_prey))
            self.assertGreaterEqual(new_prey_count, initial_prey_count + expected_new_prey)
            initial_prey_count = new_prey_count

    def test_random_actions(self):
        observations = self.env.reset()
        
        for _ in range(15):
            initial_predator_count = len([a for a in self.env.agents if "predator" in a.role])
            initial_prey_count = len([a for a in self.env.agents if "prey" in a.role])
            initial_predator_health = [a.health for a in self.env.agents if "predator" in a.role][0]

            actions = {agent.id: random.choice([1, 2, 3, 4]) for agent in self.env.agents}
            new_observations, rewards, dones = self.env.step(actions)

            self.assertEqual(len(new_observations), len(self.env.agents))

            all_agent_ids = {agent.id for agent in self.env.agents}  
            all_reward_ids = set(rewards.keys()) 
            all_done_ids = set(dones.keys())  

            self.assertTrue(all_agent_ids.issubset(all_reward_ids))  # Living agents must have rewards
            self.assertTrue(all_agent_ids.issubset(all_done_ids))  # Living agents must have dones

            new_predator_health = [a.health for a in self.env.agents if "predator" in a.role][0]
            self.assertNotEqual(new_predator_health, initial_predator_health)  # Health should change

            # Check if the number of predators and prey has changed (due to hunting or starvation)
            new_predator_count = len([a for a in self.env.agents if "predator" in a.role])
            new_prey_count = len([a for a in self.env.agents if "prey" in a.role])
            self.assertLessEqual(new_predator_count, initial_predator_count)
            self.assertLessEqual(new_prey_count, initial_prey_count)

            observations = new_observations

if __name__ == "__main__":
    unittest.main()
