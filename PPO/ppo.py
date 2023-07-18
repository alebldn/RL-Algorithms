from __future__ import annotations 

import os 
import pickle
import shutil

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym


class PPO():

    def __init__(self, 
            envs: gym.vector.VectorEnv, 
            actor_lr: float,
            critic_lr: float, 
            gamma: float,
            lam: float,
            ent_coef: float,
            clip_ratio: float,
            ppo_epochs: int,
            num_epochs: int,
            num_steps: int,
            mini_batch_size: int
            ) -> None:

        self.envs = envs
        self.n_envs = envs.num_envs
        self.state_shape = obs_shape = envs.single_observation_space.shape[0]
        self.n_actions = envs.single_action_space.n

        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.mini_batch_size = mini_batch_size

        self.build_models(n_envs, actor_lr, critic_lr)


    def build_models(self, n_envs: int, actor_lr: int, critic_lr: int) -> None:

        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()

        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr, global_clipnorm=5.0)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr, global_clipnorm=5.0)


    def build_actor_model(self) -> tf.keras.Model:
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.n_actions, activation='softmax')])


    def build_critic_model(self) -> tf.keras.Model:
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)])


    def train(self) -> [np.array, np.array, np.array]:

        # create a wrapper environment to save episode returns and episode lengths
        self.envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=self.n_envs * self.num_epochs)
        actor_losses, entropies, critic_losses, global_rewards  = [], [], [], []
        # Autoresets, so you can reset only at the beginning
        last_states, info = self.envs_wrapper.reset(seed=42)

        for epoch in tqdm(range(self.num_epochs)):

            epoch_actor_losses, epoch_entropies, epoch_critic_losses = [], [], [] 
            
            # Gather experience
            states, probs, values, actions, rewards, masks = self.rollout(last_states)
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(values, rewards, masks)

            # Perform ppo update
            for _ in range(self.ppo_epochs):
                indices = np.arange(self.num_steps)
                np.random.shuffle(indices)

                for i in range(self.num_steps // self.mini_batch_size):

                    batch_indices = indices[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
                    
                    batch_states = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
                    batch_probs = tf.convert_to_tensor(probs[batch_indices], dtype=tf.float32)
                    batch_actions = tf.convert_to_tensor(actions[batch_indices], dtype=tf.int32)
                    batch_advantages = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                    batch_returns = tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                    
                    actor_loss, entropy, critic_loss = self.perform_update(batch_states, batch_probs, batch_actions, batch_advantages, batch_returns)
                    
                    epoch_actor_losses.append(np.mean(actor_loss.numpy()))
                    epoch_entropies.append(np.mean(entropy.numpy()))
                    epoch_critic_losses.append(np.mean(critic_loss.numpy()))

            actor_losses.append(np.mean(epoch_actor_losses))
            entropies.append(np.mean(epoch_entropies))
            critic_losses.append(np.mean(epoch_critic_losses))
            global_rewards.append(np.mean(rewards))

        return np.array(actor_losses), np.array(entropies), np.array(critic_losses), np.array(global_rewards)
    
        
    def rollout(self, states: np.array) -> [np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        
        ep_states, ep_probs, ep_values, ep_actions, ep_rewards, ep_masks = [], [], [], [], [], []

        for step in range(self.num_steps):

            ep_states.append(states)
            
            probs, values, actions = self.sample_actions(tf.convert_to_tensor(states))
            probs, values, actions = probs.numpy(), values.numpy(), actions.numpy()

            states, rewards, terminated, truncated, infos = self.envs_wrapper.step(actions)
            mask = np.logical_not(np.logical_or(terminated, truncated))

            ep_probs.append(probs)
            ep_values.append(values)
            ep_actions.append(actions)
            ep_rewards.append(rewards)
            ep_masks.append(mask)
        
        bootstrap = tf.squeeze(self.critic_model(states)).numpy()
        ep_values.append(bootstrap)
        ep_states.append(states)
        
        return np.array(ep_states), np.array(ep_probs), np.array(ep_values), np.array(ep_actions), np.array(ep_rewards), np.array(ep_masks)


    @tf.function
    def sample_actions(self, states: tf.Tensor) -> [tf.Tensor, tf.Tensor, tf.Tensor]:

        probs = self.actor_model(states)
        values = tf.squeeze(self.critic_model(states), axis=-1) # Squeeze it because the resulting shape is [n_envs, 1]
        action_pd = tfp.distributions.Categorical(probs=probs)
        actions = action_pd.sample()
            
        return probs, values, actions


    def compute_advantages(self, values: np.array, rewards: np.array, masks: np.array) -> [np.array, np.array]:
        
        next_values = values[1:]
        values = values[:self.num_steps]
        next_advantage = 0.0
        advantages, returns = [], []

        for t in reversed(range(self.num_steps)):
            expected_return = rewards[t] + (self.gamma * masks[t] * next_values[t])
            delta = expected_return - values[t]
            next_advantage = delta + (self.gamma * self.lam * masks[t]) * next_advantage

            advantages.append(next_advantage)
            returns.append(expected_return)

        advantages = np.array(advantages[::-1])
        returns = np.array(returns[::-1])

        # normalize advantages and returns? 
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns


    @tf.function
    def perform_update(self, batch_states: tf.Tensor, batch_probs: tf.Tensor, batch_actions: tf.Tensor, batch_advantages: tf.Tensor, batch_returns: tf.Tensor) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(watch_accessed_variables=False) as actor_tape:
            actor_tape.watch(self.actor_model.trainable_variables)
            actor_loss, entropy = self.actor_loss(batch_states, batch_probs, batch_advantages, batch_actions)
            total_loss = actor_loss - entropy

        actor_grads = actor_tape.gradient(total_loss, self.actor_model.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as critic_tape:
            critic_tape.watch(self.critic_model.trainable_variables)
            critic_loss = self.critic_loss(batch_states, batch_returns)

        critic_grads = critic_tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        return actor_loss, entropy, critic_loss

    @tf.function
    def actor_loss(self, batch_states: tf.Tensor, batch_old_probs: tf.Tensor, batch_advantages: tf.Tensor, batch_actions: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        # batch_states with shape [mini_batch_size, n_envs, n_states]
        # batch_old_probs with shape [mini_batch_size, n_envs, n_actions]       
        # batch_advantages with shape [mini_batch_size, n_envs]
        # batch_actions with shape [mini_batch_size, n_envs] [values from 0 to n_actions-1]

        # Reshape batch states so to be able to make the prediction
        batch_states = tf.reshape(batch_states, (self.mini_batch_size * self.n_envs, self.state_shape))
       
        # Predict probabilites given by the old model and by the new, updated one
        # batch_probs with shape shape: [mini_batch_size * n_envs, n_actions]
        batch_probs = self.actor_model(batch_states, training=True)                            
        
        # Reshape it back to the original shape
        # batch_probs with shape shape: [mini_batch_size, n_envs, n_actions]
        batch_probs = tf.reshape(batch_probs, (self.mini_batch_size, self.n_envs, self.n_actions))
        batch_old_probs = tf.reshape(batch_old_probs, (self.mini_batch_size, self.n_envs, self.n_actions))

        # Make batch_actions one hot and compute the probabilities
        batch_actions = tf.one_hot(batch_actions, self.n_actions)
        batch_probs = tf.reduce_sum(batch_probs * batch_actions, axis=-1)
        batch_probs = tf.clip_by_value(batch_probs, 1e-8, 1.0)
        batch_old_probs = tf.reduce_sum(batch_old_probs * batch_actions, axis=-1)
        batch_old_probs = tf.clip_by_value(batch_old_probs, 1e-8, 1.0)

        # Convert it to log for ratio computation and clip it 
        ratio = batch_probs / batch_old_probs
        clipped_ratio = tf.clip_by_value(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)
        
        # Get the minimum advantage between ratio and clipped_ratio
        min_batch_advantage = tf.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages)
        # From https://keras.io/examples/rl/ppo_cartpole
        # min_batch_advantage = tf.where(batch_advantage > 0, (1+self.clip_ratio, 1-self.clip_ratio 
        
        # Compute loss 
        actor_loss = - tf.reduce_mean(min_batch_advantage, axis=0)
        actor_loss = tf.reduce_mean(actor_loss)

        # Compute entropy
        entropy = - self.ent_coef * tf.reduce_mean(tf.reduce_sum(batch_probs * tf.math.log(batch_probs), axis=-1), axis=0)
        entropy = tf.reduce_mean(entropy)

        return actor_loss, entropy

    @tf.function
    def critic_loss(self, batch_states: tf.Tensor, batch_returns: tf.Tensor) -> tf.Tensor:
        batch_states = tf.reshape(batch_states, (self.mini_batch_size * self.n_envs, self.state_shape))
        values = self.critic_model(batch_states, training=True)
        values = tf.reshape(values, (self.mini_batch_size, self.n_envs))
        critic_loss = tf.reduce_mean(tf.square(values - batch_returns), axis=0)
        critic_loss = tf.reduce_mean(critic_loss)

        return critic_loss

    def plot(self, global_rewards: np.array, actor_losses: np.array, entropies: np.array, critic_losses: np.array):
        print('Actor\tCritic\tEntropy')
        print('#########################################################')
        for a, c, e in zip(actor_losses, critic_losses, entropies):
            print(f'{a}\t{c}\t{e}')
        print('#########################################################')

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        fig.suptitle(f"Training plots for {agent.__class__.__name__} in the LunarLander-v2 environment\n \
                (n_envs = {n_envs}, num_steps = {num_steps}, randomize_domain = {randomize_domain})")

        # episode return
        axs[0][0].set_title("Episode Returns")
        axs[0][0].plot(global_rewards)
        axs[0][0].set_xlabel("Number of episodes")

        # actor_loss
        axs[0][1].set_title("Actor Loss")
        axs[0][1].plot(actor_losses)
        axs[0][1].set_xlabel("Number of updates")

        # entropy
        axs[1][0].set_title("Entropy")
        axs[1][0].plot(entropies)
        axs[1][0].set_xlabel("Number of updates")

        # critic_loss
        axs[1][1].set_title("Critic Loss")
        axs[1][1].plot(critic_losses)
        axs[1][0].set_xlabel("Number of updates")

        plt.tight_layout()
        plt.show()



    def play(self, play_envs: list(gym.Env)):
        for play_env in play_envs:

            ep_reward = 0
            state, info = play_env.reset()
            while True:
                state = np.expand_dims(state, axis=0)
                _, _, action = self.sample_actions(state)
                action = np.squeeze(action.numpy(), axis=0)
                state, reward, term, trunc, _ = play_env.step(action)
                play_env.render()
                ep_reward += reward

                if term or trunc:
                    print(ep_reward)
                    break



def get_env(render=False) -> gym.Env:
    return gym.make(
            "LunarLander-v2",
            gravity=np.clip(
                np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
            ),
            enable_wind=np.random.choice([True, False]),
            wind_power=np.clip(
                np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
            ),
            turbulence_power=np.clip(
                np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
            ),
            max_episode_steps=600,
            autoreset=True,
            render_mode='human' if render else 'rgb_array'
        )



# hyperparams
n_envs = 32
envs = gym.vector.AsyncVectorEnv([ lambda: get_env() for _ in range(n_envs) ])
play_envs = [get_env(render=True) for _ in range(n_envs)])

agent = PPO(
        envs=envs, 
        actor_lr=2.5e-4, 
        critic_lr=7.5e-4, 
        gamma=0.999, 
        lam=0.95, 
        ent_coef=0.01, 
        clip_ratio=0.2, 
        ppo_epochs=32, 
        num_epochs=64, 
        num_steps=1024, 
        mini_batch_size=128)

actor_losses, entropies, critic_losses, global_rewards = agent.train()

agent.play(play_envs)

envs.close()
env.close() for env in envs 
