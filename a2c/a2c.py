import sys
sys.path.append('.')

from pg_agent.pg_agent import PGAgent

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym


class A2C(PGAgent):

    def __init__(self, 
            gamma: float,
            lam: float,
            ent_coef: float,
            num_epochs: int,
            num_steps: int,
            ) -> None:

        super().__init__(gamma, lam, ent_coef, num_epochs, num_steps)


    def train_step(self, batch_states: np.array, batch_probs: np.array, batch_actions: np.array, batch_advantages: np.array, batch_returns: np.array) -> [np.array, np.array, np.array]:
        epoch_actor_losses, epoch_entropies, epoch_critic_losses = [], [], []

        batch_states = tf.convert_to_tensor(batch_states, dtype=tf.float32)
        batch_probs = tf.convert_to_tensor(batch_probs, dtype=tf.float32)
        batch_actions = tf.convert_to_tensor(batch_actions, dtype=tf.int32)
        batch_advantages = tf.convert_to_tensor(batch_advantages, dtype=tf.float32)
        batch_returns = tf.convert_to_tensor(batch_returns, dtype=tf.float32)

        actor_loss, entropy, critic_loss = self.perform_update(batch_states, batch_probs, batch_actions, batch_advantages, batch_returns)

        epoch_actor_losses.append(np.mean(actor_loss.numpy()))
        epoch_entropies.append(np.mean(entropy.numpy()))
        epoch_critic_losses.append(np.mean(critic_loss.numpy()))

        return np.mean(epoch_actor_losses), np.mean(epoch_entropies), np.mean(epoch_critic_losses)

    @tf.function
    def actor_loss(self, batch_states: tf.Tensor, batch_old_probs: tf.Tensor, batch_actions: tf.Tensor, batch_advantages: tf.Tensor) -> [tf.Tensor, tf.Tensor]:

        batch_states = tf.reshape(batch_states, (self.num_steps * self.n_envs, self.state_shape))
        batch_probs = self.actor_model(batch_states, training=True)
        batch_probs = tf.clip_by_value(batch_probs, 1e-8, 1.0)
        batch_probs = tf.reshape(batch_probs, (self.num_steps, self.n_envs, self.n_actions))

        batch_log_probs = tf.math.log(batch_probs)
        action_log_probs = tf.reduce_sum(batch_log_probs * tf.one_hot(batch_actions, self.n_actions), axis=-1)

        actor_loss = - tf.reduce_mean(batch_advantages * action_log_probs, axis=0)
        actor_loss = tf.reduce_mean(actor_loss)

        entropy = - self.ent_coef * tf.reduce_mean(tf.reduce_sum(batch_probs * batch_log_probs, axis=-1), axis=0)
        entropy = tf.reduce_mean(entropy)

        return actor_loss, entropy

    @tf.function
    def critic_loss(self, batch_states: tf.Tensor, batch_returns: tf.Tensor) -> tf.Tensor:
        batch_states = tf.reshape(batch_states, (self.num_steps * self.n_envs, self.state_shape))
        values = self.critic_model(batch_states, training=True)
        values = tf.reshape(values, (self.num_steps, self.n_envs))

        critic_loss = tf.reduce_mean(tf.square(values - batch_returns), axis=0)
        critic_loss = tf.reduce_mean(critic_loss)

        return critic_loss

