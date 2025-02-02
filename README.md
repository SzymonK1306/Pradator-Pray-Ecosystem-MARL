# 🧫 Predator-Prey Ecosystem using Multi-Agent RL

### 📑 About

This repository presents a predator-prey environment featuring large-scale multi-agent training (up to 2000 agents). The main aim is to simulate and study predator-prey relationships similar to the Lotka-Volterra model. Each agent learns its strategy and adapts to a constantly evolving environment.

### 🧬 Environments

We use a grid-based setup, where each cell can contain walls, prey, or predators. Three variants were introduced:
- Type 1. Basic environment with default rewards and penalties
- Type 2. Similar setup with additional constraints, leading to more complex population dynamics 
- Type 3. Agents have genetic traits that are inherited and recombined in each new generation, influencing survival and adaptation

The following gif presents rendered environment during several epochs training. Prey are red, predators are blue, walls are black, the hunting area has a light green background.

<img src="_readme-img/env.gif?raw=true" width="400" alt="Example environment">

### 🧠 Double Deep Q-Learning

Our primary focus is Double Deep Q-Learning (DDQN), which splits the Q-value estimation into policy and target networks for enhanced training stability.  

<img src="_readme-img/model.png?raw=true" width=400 alt="DDQN">

### 📈 Results

- DDQN typically adapts well to rapid changes in multi-agent settings, maintaining competitiveness among a large number of predators and prey.  
- We also tested PPO (Proximal Policy Optimization) in limited trials. Due to stricter policy update constraints, it struggled to adapt swiftly in highly dynamic environments.  
- The cyclic nature of predator-prey populations was generally observed, albeit with minor deviations linked to environmental parameters.  

Additional plots and detailed logs are available upon request.

### 💡 Acknowledgements

This project was inspired by the following article. We extend special thanks to the researchers whose insights laid the groundwork for our experiments.

> Yamada, J., Shawe-Taylor, J., & Fountas, Z. (2020, July). Evolution of a complex predator-prey ecosystem on large-scale multi-agent deep reinforcement learning. In 2020 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.
