from ppo import PPO
import numpy as np
import gymnasium as gym

# As bad as it looks, using the "render_mode='human' if play else 'rgb_array' trick makes every step of the training last way longer
def get_env(play=False) -> gym.Env:
    if play:
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
                render_mode='human'
            )

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
    )


# hyperparams
n_envs = 32
envs = gym.vector.AsyncVectorEnv([ lambda: get_env() for _ in range(n_envs) ])
# envs = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=self.n_envs * self.num_epochs)
play_envs = [get_env(play=True) for _ in range(16)]

agent = PPO(
        gamma=0.999, 
        lam=0.95, 
        ent_coef=0.01, 
        num_steps=1024, 
        clip_ratio=0.2, 
        ppo_epochs=32, 
        num_epochs=8, 
        mini_batch_size=128)

agent.set_vector_env(envs) 
agent.set_models(actor_lr=2.5e-4, critic_lr=7.5e-4)
mean_rewards, actor_losses, entropies, critic_losses = agent.train()
agent.plot(mean_rewards, actor_losses, entropies, critic_losses)
agent.play(play_envs)
envs.close()
for env in play_envs:
    env.close()
