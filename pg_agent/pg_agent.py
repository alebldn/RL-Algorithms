from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym

""" 
    Main class for a policy gradient agent. It builds two neural networks, i.e.: agent and critic to act and learn how to maximize the rewards obtained by exploring the environments. 
    This class has to be extended by an actual policy gradient method (see REINFORCE, Actor-Critic, PPO).  
    Aside from building actor and critic neural networks (separately), it gives standard functions to be used in sublasses but it requires a definition of some functions:
    - train_step:           Function that is called in the final stages of every epoch and is the function needed to preprocess all the data obtained by rollout() during environment exploration. 
                            The function implemented in the subclass *must* invoke perform_update to train the neural networks by computing the actor and the critic loss by invoking respectively actor_loss and critic_loss.
    - actor_loss:           Function called in perform_update that computes actor model's loss according to the experience gathered in rollout.
    - critic_loss:          Function called in perform_update that computes critic model's loss according to the experience gathered in rollout.

"""

class PGAgent():
    """
    Inputs: 
        - gamma:            Expected rewards' coefficient for continuous environments.
        - lam:              Lambda, the exponential coefficient for computing Generalized Advantage Estimation.
        - ent_coef:         Entropy coefficient to introduce noise in the actor loss so to avoid overfitting to a single solution, hence finding only local maxima.
        - num_epochs:       Amount of epochs to be used to train the agent.
        - num_steps:        Number of steps performed in the environment at each epoch.
    """ 
    def __init__(self, 
            gamma: float,
            lam: float,
            ent_coef: float,
            num_epochs: int,
            num_steps: int,
            ) -> None:

        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.num_epochs = num_epochs
        self.num_steps = num_steps


    """
    set_vector_env:
    Set the vector of to train the agent on. 
    Inputs: 
        - envs:             VectorEnv containing the environments. Can be either VectorEnv or any of its subclasses like AsyncVectorEnv
    """
    def set_vector_env(self, envs: gym.vector.VectorEnv) -> None:
        self.envs = envs
        self.n_envs = envs.num_envs
        self.state_shape = obs_shape = envs.single_observation_space.shape[0]
        self.n_actions = envs.single_action_space.n


    """
    set_models:
    Automatically set neural networks and optimizers.
    Inputs:
        - actor_lr:         Float that represents the actor model's learning rate.
        - critic_lr:        Float that represents the critic model's learning rate.

    """
    def set_models(self, actor_lr=1e-3: float, critic_lr=1e-3: float) -> None:
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
        
    

    """
    train:
    Main loop function to train the model. For each epoch performs the cycle observe, think, act via the rollout function and calls train_step to learn on gathered experience.
    Outputs:
        - mean_rewards:         Numpy array containing the mean rewards obtained during training.           Shape: [num_epochs]
        - actor_losses:         Numpy array containing the actor losses computed during training.           Shape: [num_epochs]
        - critic_losses:        Numpy array containing the critic losses computed during training.          Shape: [num_epochs]
        - entropies:            Numpy array containing the entropies computed during training.              Shape: [num_epochs]
    """
    def train(self) -> [np.array, np.array, np.array, np.array]:
        actor_losses, entropies, critic_losses, mean_rewards = [], [], [], []
        last_states, info = self.envs.reset(seed=42)

        for epoch in tqdm(range(self.num_epochs)):

            # Gather experience
            states, probs, values, actions, rewards, masks = self.rollout(last_states)
            last_states = states[-1]
            states = states[:-1]
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(values, rewards, masks)
            
            # Train step
            epoch_actor_losses, epoch_entropies, epoch_critic_losses = self.train_step(states, probs, actions, advantages, returns)

            actor_losses.append(epoch_actor_losses)
            entropies.append(epoch_entropies)
            critic_losses.append(epoch_critic_losses)
            mean_rewards.append(np.mean(rewards))

        return np.array(mean_rewards), np.array(actor_losses), np.array(entropies), np.array(critic_losses)


    """
    rollout:
    Function to explore the environment array. For num_steps steps this function gathers experience by observing the environment, thinking about the best action to perform, and performing said action.
    Output states and values have an unusual shape since they contain respectively the last states obtained by performing the step and the bootstrap. 
    Inputs: 
        - states:           Numpy array containing the states either obtained by resetting the 
                            environments (the very first epoch) or from the previous rollout call.          Shape: [state_shape]
    Outputs:
        - ep_states:        Numpy array containing the states traversed.                                    Shape: [num_steps + 1, n_envs, [state_shape]]
        - ep_probs:         Numpy array containing the  probabilities obtained from the actor network
                            for later usage in algorithms like ppo.                                         Shape: [num_steps, n_envs, n_actions]
        - ep_values:        Numpy array containing the values obtained from the critic network.             Shape: [num_steps + 1, n_envs]
        - ep_actions:       Actions performed in this epoch.                                                Shape: [num_steps, n_envs]
        - ep_rewards:       Rewards obtained in this epoch.                                                 Shape: [num_steps, n_envs] 
        - ep_masks:         Masks to be used for Generalized Advantages Estimation for continuous tasks.    
                            Used to indidcate that the episode has not been terminated nor truncated.       Shape: [num_steps, n_envs]
    """
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


    """
    sample_actions:
    Given an array of states, sample an action given probabilities obtained from the actor model. Uses @tf.function for better performances in computing probs and values.
    Inputs:
        - states:           Tensor containing the states given in inputs to the networks.                   Shape: [state_shape]
    Outputs:
        - probs:            Tensor containing the probabilities obtained from the actor neural network.     Shape: [num_steps, n_envs, n_actions]
        - values:           Tensor containing the values obtained from the critic network.                  Shape: [num_steps, n_envs] 
        - actions:          Tensor containing the action performed (as an integer, not one hot encoded).    Shape: [num_steps, n_envs]
    """
    @tf.function
    def sample_actions(self, states: tf.Tensor) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        
        probs = self.actor_model(states)
        values = tf.squeeze(self.critic_model(states), axis=-1) # Squeeze it because the resulting shape is [n_envs, 1]
        action_pd = tfp.distributions.Categorical(probs=probs)
        actions = action_pd.sample()
            
        return probs, values, actions


    """
    compute_advantages:
    Computes the advantages using the Generalized Advantage Estimation technique. 
    Inputs:
        - values:       Numpy array containing the values obtained in rollout.                              Shape: [num_steps + 1, n_envs]
        - rewards:      Numpy array containing the rewards obtained in rollout.                             Shape: [num_steps, n_envs]
        - masks:        Numpy array containing the masks obtained in rollout.                               Shape: [num_steps, n_envs]    
    Outputs: 
        - advantages:   Numpy array containing the advantages to be used to train actor model.              Shape: [num_steps, n_envs]
        - returns:      Numpy array containing the expected returns to be used to train critic model.       Shape: [num_steps, n_envs]
    """
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


    """ 
    perform_update:
    Updates the agent neural networks by calling the subclass' agent_loss and critic_loss functions. Every paramter's definition is quite trivial except for the fact that these are batched parameters prepared in the train_step function that has to be implemented in the subclass. This function is useful for those algorithms like ppo that require an intermediate step for batching data and shuffling it. Standard algorithms like Actor Critic that do not require batching simply have batch_variable = variable. Again, using tf.function for efficiency in tensor computation.
    Inputs:
        - batch_states:     Tensor containing the states possibly in a batched form.                        Shape: dependant on train_step.
        - batch_probs:      Tensor containing the original probabilities possibly in a batched form.        Shape: dependant on train_step.
        - batch_actions:    Tensor containing the actions performed possibly in a batched form.             Shape: dependant on train_step.
        - batch_advantages: Tensor containing the advantages possibly in a batched form.                    Shape: dependant on train_step.
        - batch_returns:    Tensor containing the expected returns possibly in a batched form.              Shape: dependant on train_step.
    Outputs:
        - actor_loss:       Tensor containing the actor loss.                                               Shape: [1]
        - entropy:          Tensor containing the entropy.                                                  Shape: [1]
        - critic_loss:      Tensor containing the critic loss.                                              Shape: [1]
    """
    @tf.function
    def perform_update(self, batch_states: tf.Tensor, batch_probs: tf.Tensor, batch_actions: tf.Tensor, batch_advantages: tf.Tensor, batch_returns: tf.Tensor) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(watch_accessed_variables=False) as actor_tape:
            actor_tape.watch(self.actor_model.trainable_variables)
            actor_loss, entropy = self.actor_loss(batch_states, batch_probs, batch_actions, batch_advantages)
            total_loss = actor_loss - entropy

        actor_grads = actor_tape.gradient(total_loss, self.actor_model.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as critic_tape:
            critic_tape.watch(self.critic_model.trainable_variables)
            critic_loss = self.critic_loss(batch_states, batch_returns)

        critic_grads = critic_tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        return actor_loss, entropy, critic_loss


    """
    plot:
    Plots the data obtained during training. 
    Inputs:
        - mean_rewards:     Numpy array containing the mean rewards.                                        Shape: [num_epochs]
        - actor_losses:     Numpy array containing the actor losses.                                        Shape: [num_epochs]
        - entropies:        Numpy array containing the entropies.                                           Shape: [num_epochs]
        - critic_losses:    Numpy array containing the critic losses.                                       Shape: [num_epochs]
    """
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


    """
    play:
    Given a list of environments (that might or might not be the same as the environments used for training) plays and displays the screen for each of them.
    Inputs:
        - play_envs:        list containing environments for the agent to play on.
    """
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

