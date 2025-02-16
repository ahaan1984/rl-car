# Autonomous Racing AI

This project implements an autonomous car racing AI using deep reinforcement learning. The AI agent learns to navigate a procedurally generated racetrack using a spatially aware neural network.

## Features
- Procedural track generation using convex hulls and splines
- Reinforcement learning using a deep Q-network (DQN)
- Car physics simulation with sensor-based perception
- Environment and agent interaction for training and evaluation

## Installation
### Prerequisites
Ensure you have Python installed (>= 3.8).

### Install Dependencies
This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

1. Install `uv` (if not already installed):
   ```sh
   pip install uv
   ```

2. Create a virtual environment and install dependencies:
   ```sh
   uv venv .venv
   uv add numpy scipy torch pygame
   ```

3. Activate the virtual environment:
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```

## Usage
### Training the Agent
Run the following command to start training the reinforcement learning agent:
```sh
python main.py
```
The trained model will be saved as `final_dqn_agent.pth`.

### Running the Environment
You can visualize the racetrack and car movement by executing:
```sh
python environment.py
```

## Project Structure
```
.
├── agent.py         # Deep reinforcement learning agent (DQN)
├── car.py           # Car physics and control logic
├── environment.py   # Game environment with racetrack
├── main.py          # Training script for AI agent
├── track.py         # Track generation and rendering
├── requirements.txt # Project dependencies
└── README.md        # Project documentation
```

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `pygame`
- `torch`
- `scipy`

## License
This project is open-source and available under the MIT License.

