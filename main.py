import yaml
from simple_environment import Mars_Environment
from deep_q_environment import Deep_Q_Mars


with open("config.yaml", "r") as f:
    configs = yaml.safe_load(f)

## Simple Q-Learning
mars_environment = Mars_Environment(
    config = configs["5x5_simple_game"],
    max_epsilon = 1,
    min_epsilon = 0.005,
    epsilon_decay_rate = 0.0001,
    alpha = 0.7,
    gamma = 0.4,
    no_episodes = 10000,
    max_steps = 100,
    policy="softmax",
    max_temperature=100,
    min_temperature=10,
    temperature_decay_rate=0.00005,
    save_results=True,
    display=True
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

