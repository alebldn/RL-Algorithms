from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym

class PGAgent():

    def __init__(self, 
            gamma: float,
            lam: float,
            ent_coef: float,
            num_steps: int,
            ) -> None:

        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.num_steps = num_steps


    def set_vector_env(self, envs: gym.vector.VectorEnv) -> None:
        self.envs = envs
        self.n_envs = envs.num_envs
        self.state_shape = obs_shape = envs.single_observation_space.shape[0]
        self.n_actions = envs.single_action_space.n



    def set_models(self, actor_lr: int, critic_lr: int) -> None:
        self.actor_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.n_actions, activation='softmax')])
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr, global_clipnorm=5.0)
        self.critic_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)])
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr, global_clipnorm=5.0)
        
        
    def train(self) -> [np.array, np.array, np.array]:
        actor_losses, entropies, critic_losses, mean_rewards = [], [], [], []
        last_states, info = self.envs.reset(seed=42)

        for epoch in tqdm(range(self.num_epochs)):

            # Gather experience
            states, probs, values, actions, rewards, masks = self.rollout(last_states)
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(values, rewards, masks)
            
            # Train step
            epoch_actor_losses, epoch_entropies, epoch_critic_losses = self.train_step(states, probs, actions, advantages, returns)

            actor_losses.append(epoch_actor_losses)
            entropies.append(epoch_entropies)
            critic_losses.append(epoch_critic_losses)
            mean_rewards.append(np.mean(rewards))

        return np.array(mean_rewards), np.array(actor_losses), np.array(entropies), np.array(critic_losses)


    def rollout(self, states: np.array) -> [np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        ep_states, ep_probs, ep_values, ep_actions, ep_rewards, ep_masks = [], [], [], [], [], []

        for step in range(self.num_steps):
            ep_states.append(states)
            
            probs, values, actions = self.sample_actions(tf.convert_to_tensor(states))
            probs, values, actions = probs.numpy(), values.numpy(), actions.numpy()
            states, rewards, terminated, truncated, infos = self.envs.step(actions)
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


    def plot(self, mean_rewards: np.array, actor_losses: np.array, entropies: np.array, critic_losses: np.array):
        print('Mean Rewards | Actor | Critic | Entropy')
        print('#########################################################')
        for r, a, c, e in zip(mean_rewards, actor_losses, critic_losses, entropies):
            print(f'{r}\t{a}\t{c}\t{e}')
        print('#########################################################')

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        fig.suptitle(f"Training plots for {self.__class__.__name__}")

        # episode return
        axs[0][0].set_title("Mean Rewards")
        axs[0][0].set_xlabel("Number of episodes")
        axs[0][0].plot(mean_rewards)

        # actor_loss
        axs[0][1].set_title("Actor Loss")
        axs[0][1].set_xlabel("Number of updates")
        axs[0][1].plot(actor_losses)

        # entropy
        axs[1][0].set_title("Entropy")
        axs[1][0].set_xlabel("Number of updates")
        axs[1][0].plot(entropies)

        # critic_loss
        axs[1][1].set_title("Critic Loss")
        axs[1][0].set_xlabel("Number of updates")
        axs[1][1].plot(critic_losses)

        plt.tight_layout()
        plt.show()


    def play(self, play_envs: list):
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

