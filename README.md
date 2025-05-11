# ppo.py Documentation

This script implements a basic Proximal Policy Optimization (PPO) algorithm for reinforcement learning. The code is structured to allow training, resuming training, and evaluating a trained agent on various OpenAI Gym environments.

## Code Structure

1. **Imports and Dependencies**:
    - The script uses PyTorch for neural network implementation and optimization.
    - It uses OpenAI Gym for the environment interface.
    - TensorBoard is used for logging training metrics.

2. **NeuralNetwork Class**:
    - A simple feedforward neural network with two hidden layers using Tanh activation.
    - Used for both the actor (policy network) and the critic (value function network).

3. **PPO Class**:
    - Implements the PPO algorithm.
    - Key methods:
      - `__init__`: Initializes the environment, actor, critic, optimizers, and hyperparameters.
      - `action_sample`: Samples an action from the policy distribution.
      - `learn`: Trains the agent using PPO-Clip objective and critic loss.
      - `run`: Evaluates the trained agent in the environment.
      - `__delete__`: Cleans up resources (closes environment and TensorBoard writer).

4. **Main Function**:
    - Parses command-line arguments to determine the mode of operation (train or evaluate).
    - Creates an instance of the PPO class and calls the appropriate method (`learn` for training, `run` for evaluation).

## How to Train from Scratch

1. Open a terminal and navigate to the directory containing `ppo.py`.
2. Run the following command:
    ```bash
    python ppo.py --train --env <ENV_NAME>
    ```
    Replace `<ENV_NAME>` with the desired Gym environment (e.g., `Pendulum-v1`, `MountainCarContinuous-v0`, `LunarLander-v3`).

3. The training process will begin, and the model will save the best-performing actor and critic networks to disk (`ppo_actor_<ENV_NAME>.pth` and `ppo_critic_<ENV_NAME>.pth`).

4. Training will stop automatically if the moving average of rewards stabilizes for a specified number of iterations.

## How to Resume Training

1. Ensure the saved model files (`ppo_actor_<ENV_NAME>.pth` and `ppo_critic_<ENV_NAME>.pth`) are in the same directory as `ppo.py`.
2. Run the following command:
    ```bash
    python ppo.py --train --resume --env <ENV_NAME>
    ```
    The script will load the saved models and continue training from the last checkpoint.

## How to Perform Inference (Evaluation)

1. Ensure the trained model file (`ppo_actor_<ENV_NAME>.pth`) is in the same directory as `ppo.py`.
2. Run the following command:
    ```bash
    python ppo.py --env <ENV_NAME>
    ```
    The agent will load the trained actor model and evaluate its performance in the specified environment. The total reward for the evaluation episode will be printed to the console.