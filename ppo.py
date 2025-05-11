# This is a simple PPO implementation, trying to implement basic algorithm from the PPO paper

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
    def __init__(self, train:bool):

        self.env_name = "Pendulum-v1" #"Pendulum-v1",""MountainCarContinuous-v0""
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
            self.k_iterations = 3500
            self.d_trajectory = 10
            self.t_steps = 250
            
            self.gamma = 0.95
            self.clip = 0.2
            
            self.update_runs = 5
            
            self.save_every = 10
            
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
            self.log_write = SummaryWriter(filename_suffix=self.env_name)
    
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
            
            print(f"{k}/{self.k_iterations} reward {total_reward_iteration} act_loss {actor_loss} crtc_loss {critic_loss}")
            
            self.log_write.add_scalar("loss/actor", actor_loss, k)
            self.log_write.add_scalar("loss/critic", critic_loss, k)
            self.log_write.add_scalar("rewards", total_reward_iteration, k)
            
            if k % self.save_every == 0:
                torch.save(self.actor.state_dict(), self.actor_dict_path)
                torch.save(self.critic.state_dict(), self.critic_dict_path)
    
    def run(self):
        self.actor.load_state_dict(torch.load(self.actor_dict_path))
        self.actor.eval()
        done = False
        observation, _ = self.env.reset()
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
    parser.add_argument("--train", action="store_true", help="Evaluate the agent")
    parser.add_argument("--resume", action="store_true", help="Start training from the last checkpoint")
    args = parser.parse_args()

    if args.train:
        agent = PPO(True)
        agent.learn()
    else:
        agent = PPO(False)
        agent.run()

if __name__ == "__main__":
    main()