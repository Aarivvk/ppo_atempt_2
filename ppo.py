# This is a simple PPO implementation, trying to implement basic algorithm from the PPO paper

from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import Adam
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

import argparse
import random


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_nn=64):
        super(NeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_nn),
            nn.Tanh(),
            nn.Linear(hidden_nn, hidden_nn),
            nn.Tanh(),
            nn.Linear(hidden_nn, out_dim)
        )
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        return self.network(obs)

class PPO:
    def __init__(self, train:bool, env_name:str):
        print(env_name)
        self.env_name = env_name #"Pendulum-v1","MountainCarContinuous-v0"
        self.env = gym.make(self.env_name, render_mode=None if train else "human")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        hidden_nn = 64
        self.actor = NeuralNetwork(self.obs_dim, self.act_dim, hidden_nn)
        self.actor_dict_path = f"./ppo_actor_{self.env_name}.pth"
        
        self.seed = 144
        self.env.action_space.seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if train:
            self.k_iterations = 10000
            self.d_trajectory = 10
            self.t_steps = 1250
            
            # For MountainCarContinuous-v0 0.999 and for Pendulum-v1 0.95
            if self.env_name == "MountainCarContinuous-v0":
                self.gamma = 0.999
            else:
                self.gamma = 0.95
            self.clip = 0.2
            
            self.update_runs = 5
            
            self.reward_window = deque(maxlen=50)
            self.rewards_moving_avg = -float("inf")
            self.prev_moving_avg = 0
            self.stable_count = 0
            self.saturation_threshold = 1.0  # Small change allowed between moving averages
            self.saturation_patience = 100   # How many iterations of "stable" performance before stopping

            
            # Step 1 initialize the policy and critic network
            lr = 1e-4
            
            print(self.actor)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
            self.actor_co_var = torch.full(size=(self.act_dim,), fill_value=0.5)
            self.actor_co_mat = torch.diag(self.actor_co_var)
            self.critic = NeuralNetwork(self.obs_dim, 1, hidden_nn)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
            self.critic_dict_path = f"./ppo_critic_{self.env_name}.pth"
            
            
            # Log setup to tensor board
            self.log_write = SummaryWriter(log_dir="runs", comment=self.env_name)
    
    def action_sample(self, observation):
        action_mean = self.actor(observation)
        # TODO : try with learnable covariance
        distribution = MultivariateNormal(action_mean, self.actor_co_mat)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.detach().numpy(), log_probability.detach()
        
        
        
    def learn(self):
        print("learning begins!!")
        print(f"Total iterations K 0,1,2...,{self.k_iterations}")
        for k in range(self.k_iterations):
            # Step 2 iterations
            tj_obs = []
            tj_acts = []
            tj_log_probabilities = []
            tj_rewards = []
            # Step 3 Collect set of trajectories
            for d in range(self.d_trajectory):
                observation, _ = self.env.reset(seed=self.seed)
                t_rewards =  []
                for t in range(self.t_steps):
                    tj_obs.append(observation)
            
                    action, log_probabilities = self.action_sample(observation)
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    
                    end_of_trajectory = terminated | truncated
                    
                    t_rewards.append(reward)
                    tj_acts.append(action)
                    tj_log_probabilities.append(log_probabilities)
                    
                    if end_of_trajectory:
                        break
            
                tj_rewards.append(t_rewards)
            
            tj_obs = torch.tensor(np.array(tj_obs), dtype=torch.float)
            tj_acts = torch.tensor(np.array(tj_acts), dtype=torch.float)
            tj_log_probabilities = torch.tensor(np.array(tj_log_probabilities), dtype=torch.float)
            
            # Step 4 Compute reward to go
            total_reward_iteration = 0
            tj_rtg = []
            for t_rewards in reversed(tj_rewards):
                expected_reward = 0
                for reward in reversed(t_rewards):
                    total_reward_iteration += reward
                    expected_reward = reward + expected_reward * self.gamma
                    tj_rtg.insert(0, expected_reward)
            tj_rtg = torch.tensor(np.array(tj_rtg), dtype=torch.float)
            
            # Step 5 Compute Advantage Estimates
            tj_critic_values = self.critic(tj_obs).detach()
            tj_advantages = tj_rtg - tj_critic_values
            tj_advantages = (tj_advantages - tj_advantages.mean())/(tj_advantages.std() + 1e-8)
            
            for _ in range(self.update_runs):
                # Step 6 update the policy, by optimizing PPO-Clip objective
                actions = self.actor(tj_obs)
                distribution = MultivariateNormal(actions, self.actor_co_mat)
                current_log_probabilities = distribution.log_prob(tj_acts)
                ratios = torch.exp(current_log_probabilities - tj_log_probabilities)
                s1 = ratios * tj_advantages
                s2 = torch.clamp(ratios, 1 - self.clip, 1+ self.clip) * tj_advantages
                actor_loss = (-torch.min(s1, s2)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Step 7 Fit value function / critic by regression on mean-squared error
                tj_critic_values = self.critic(tj_obs)
                critic_loss = F.mse_loss(tj_critic_values.squeeze(), tj_rtg)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            
            self.reward_window.append(total_reward_iteration)
            moving_avg = np.mean(self.reward_window)
            
            if moving_avg > self.rewards_moving_avg:
                torch.save(self.actor.state_dict(), self.actor_dict_path)
                torch.save(self.critic.state_dict(), self.critic_dict_path)
                self.rewards_moving_avg = moving_avg
                save = f" --> Saved [New Moving Avg: {moving_avg:.2f}]"
            else:
                save =""
            # Detect saturation
            if abs(moving_avg - self.prev_moving_avg) < self.saturation_threshold:
                self.stable_count += 1
            else:
                self.stable_count = 0  # reset if there's meaningful change
            self.prev_moving_avg = moving_avg
            print(f"{k}/{self.k_iterations} reward {total_reward_iteration} critic_loss {critic_loss} Stable count {self.stable_count}{save}")
            self.log_write.add_scalar("log/critic loss", critic_loss, k)
            self.log_write.add_scalar("log/rewards", total_reward_iteration, k)
            self.log_write.add_scalar("log/mv average", moving_avg, k)
            
            # Stop training if saturated
            if self.stable_count >= self.saturation_patience:
                print(f"\nTraining stopped: reward moving average stable for {self.saturation_patience} iterations.")
                break
    
    def run(self):
        self.actor.load_state_dict(torch.load(self.actor_dict_path))
        self.actor.eval()
        done = False
        observation, _ = self.env.reset(seed=self.seed)
        total_reward = 0
        while not done:
            action = self.actor(observation).detach().numpy()
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated | truncated
        print(f"evaluation total reward {total_reward}")
    
    def __delete__(self):
        self.env.close()
        self.log_write.close()


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the PPO agent.")
    parser.add_argument('--env', type=str, default="Pendulum-v1", help="Gym environment name")
    parser.add_argument("--train", action="store_true", help="Evaluate the agent")
    parser.add_argument("--resume", action="store_true", help="Start training from the last checkpoint")
    args = parser.parse_args()

    if args.train:
        agent = PPO(True, args.env)
        agent.learn()
    else:
        agent = PPO(False, args.env)
        agent.run()

if __name__ == "__main__":
    main()