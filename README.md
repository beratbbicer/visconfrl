# visconfrl
Repository for the research titled 'Integrating Visual Confidence Cues in Reinforcement Learning for Enhanced Maze Navigation: A Study on Human-Guided Autonomous Decision-making'

This study focuses on investigating the role of affective signals in action planning for autonomous agents. Specifically, it explores how a person's perceived confidence can impact the decision-making process of reinforcement learning-based autonomous agents, especially in maze navigation.

Authors: Berat Biçer, E.Batuhan Kaynak, Hamdi Dibeklioğlu

## To-Do List

### Preparatory Phase
- [ ] **Literature Review**
  - [ ] Collect papers related to affective signals and reinforcement learning.
  - [ ] Summarize findings for reference in the "Related Work" section.
  
- [x] **Tool Setup**
  - [x] Install PyTorch for deep learning tasks.
  - [x] Set up gym environment for maze simulations.

### Data Collection
- [ ] **Video Data**
  - [ ] Record short video clips of human subjects with visible faces.
  - [ ] Store the video data in a persistant storage.

- [ ] **Labeling**
  - [ ] Extract patient labels via speech and link them to perceived the grids.

### Environment Simulation
- [x] **Maze Design**
  - [x] Create multiple 8x8 grid mazes.
  - [x] Ensure each maze has a solvable path.
  
- [x] **Grid Views**
  - [x] Generate random 3x3 subregions from mazes as grid views.
  
### Model Training
- [ ] **Reinforcement Learning Agent**
  - [x] Train agent without human intervention as a control.
    - [x] Using the whole 8x8 maze.
    - [x] Using the restricted 3x3 grid.
  - [ ] Train agent with human-provided suggested actions.
    - [ ] Include this information from the start.
    - [ ] Include this information at a later stage. (Extra)

- [ ] **Confidence Model**
  - [ ] Train a computer vision model to process video clips and output confidence levels.
  
- [ ] **Joint Training**
  - [ ] Pair suggested actions with corresponding grid views.
  - [ ] Implement reinforcement scheme for agent using a deep Q-network.
  
### Evaluation
- [ ] **Performance Metrics**
  - [ ] Choose metrics for evaluating agent performance (e.g., time to solve maze, accuracy).

- [ ] **Ablation Study**
  - [ ] Compare agent trained with and without human intervention.
  
### Reporting
- [ ] **Progress Report**
  - [ ] Complete data collection, environment setup, and maze design.
  
- [ ] **Final Report**
  - [ ] Include findings, methodology, and evaluation.
