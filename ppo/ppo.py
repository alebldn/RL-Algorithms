import sys
sys.path.append('.')

from pg_agent.pg_agent import PGAgent

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym

"""
Proximal Policy Optimization
""" 
class PPO(PGAgent):

    """
    Inputs same as pg_gradient, with the addition of:
        - clip_ratio:       Clipping parameter necessary to compute the actor loss in ppo.
        - ppo_epochs:       Number of successive ppo updates using random batches for each update.
        - mini_batch_size:  Size of the mini batch in which each parameter of actor loss is split into.
    """
    def __init__(self, 
            gamma: float,
            lam: float,
            ent_coef: float,
            num_epochs: int,
            num_steps: int,
            clip_ratio: float,
            ppo_epochs: int,
            mini_batch_size: int
            ) -> None:

        super().__init__(gamma, lam, ent_coef, num_epochs, num_steps)

        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size


    """
    train_step:
    This function is used to generalize the preprocessing of perform_update by only giving it the required batches. In a2c this is only a wrapper to properly convert numpy array to tensorflow tensors.
    Input:
        - batch_states:     States explored.                                                                    Shape: [num_steps, n_envs, [state_shape]]
        - batch_probs:      Old probabilities gathered from rollout.                                            Shape: [num_steps, n_envs, n_actions]]
        - batch_actions:    Actions performed.                                                                  Shape: [num_steps, n_envs]
        - batch_advantages: Advantages computed via Generalized Advantage Estimation.                           Shape: [num_steps, n_envs]
        - batch_returns:    Returns computed via Generalized Advantage Estimation.                              Shape: [num_steps, n_envs]
    Output:
        - ep_actor_loss:    Mean of the actor loss computed in this train step.
        - ep_entropies:     Mean of the entropies computed in this train step.
        - ep_critic_loss:   Mean of the critic losses computed in this train step.
    """
    def train_step(self, states: np.array, probs: np.array, actions: np.array, advantages: np.array, returns: np.array) -> [np.array, np.array, np.array]:

        epoch_actor_losses, epoch_entropies, epoch_critic_losses = [], [], [] 
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

        return np.mean(epoch_actor_losses), np.mean(epoch_entropies), np.mean(epoch_critic_losses)


    """
    actor_loss:
    Compute the actor loss.
    Input:
        - batch_states:     States explored.
        - batch_old_probs:  Probabilities gathered during rollout (useless here, just for compatibility).
        - batch_actions:    Actions performed.
        - batch_advantages: Advantages computed via Generalized Advantage Estimation.
    Output:
        - actor_loss:       Actor loss computed.
        - entropy:          Entropy computed.
    """
    @tf.function
    def actor_loss(self, batch_states: tf.Tensor, batch_old_probs: tf.Tensor, batch_actions: tf.Tensor, batch_advantages: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
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
        # min_batch_advantage = tf.where(batch_advantage > 0, (1+self.clip_ratio, 1-self.clip_ratio) # ?!?
        
        # Compute loss 
        actor_loss = - tf.reduce_mean(min_batch_advantage, axis=0)
        actor_loss = tf.reduce_mean(actor_loss)

        # Compute entropy
        entropy = - self.ent_coef * tf.reduce_mean(tf.reduce_sum(batch_probs * tf.math.log(batch_probs), axis=-1), axis=0)
        entropy = tf.reduce_mean(entropy)

        return actor_loss, entropy

    
    """
    critic_loss:
    Compute the critic loss.
    Input:
        - batch_states:     States explored.
        - batch_returns:    Expected returns computed via Generalized Advantage Estimation.
    Output:
        - critic_loss:      Critic loss computed.
    """
    @tf.function
    def critic_loss(self, batch_states: tf.Tensor, batch_returns: tf.Tensor) -> tf.Tensor:
        batch_states = tf.reshape(batch_states, (self.mini_batch_size * self.n_envs, self.state_shape))
        values = self.critic_model(batch_states, training=True)
        values = tf.reshape(values, (self.mini_batch_size, self.n_envs))
        critic_loss = tf.reduce_mean(tf.square(values - batch_returns), axis=0)
        critic_loss = tf.reduce_mean(critic_loss)

        return critic_loss

