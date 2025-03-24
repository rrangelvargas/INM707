import yaml
from simple_environment import Mars_Environment
from deep_q_environment import Deep_Q_Mars


with open("config.yaml", "r") as f:
    configs = yaml.safe_load(f)

## Simple Q-Learning
mars_environment = Mars_Environment(
    config = configs["5x5_simple_game"],
    max_epsilon = 1,
    min_epsilon = 0.05,
    epsilon_decay_rate = 0.0005,
    alpha = 0.9,
    gamma = 0.9,
    no_episodes = 4000,
    max_steps = 300,
    policy="epsilon_greedy",
    max_temperature=100,
    min_temperature=10,
    temperature_decay_rate=0.00005,
    save_results=True,
    display=False
)

mars_environment.run()


# ## Deep Q-Learning
# deep_q_mars = Deep_Q_Mars(
#      configs["5x5_deep_q_game"],
#      no_episodes=1000,
#      max_steps=150,
#      epsilon=0.99,
#      decay=0.997,
#      min_epsilon=0.001
#  )

# deep_q_mars.run()

