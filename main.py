import yaml
from simple_environment import Mars_Environment
from deep_q_environment import MarsEnv, DeepQAgent
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    configs = yaml.safe_load(f)

def train_model(config, run_name, display=False):
    print('----------------------------------------------------')
    print(f'Running {run_name}...')
    print("Double DQN: ", config["params"]["double_dqn"])
    print("Prioritised: ", config["params"]["prioritised"])
    print('----------------------------------------------------')

    env = MarsEnv(config)
    agent = DeepQAgent(env, display=display, params=config["params"])
    agent.train(config["params"], run_name, save_graphs=True)
    agent.save_results(run_name, config["params"]["num_episodes"], config["params"]["save_checkpoint"])
    return agent

def test_model(config, run_name, display=False):
    print('----------------------------------------------------')
    print(f'Testing {run_name}...')
    print("Double DQN: ", config["params"]["double_dqn"])
    print("Prioritised: ", config["params"]["prioritised"])
    print('----------------------------------------------------')

    num_episodes = config["params"]["num_episodes"]

    env = MarsEnv(config)
    agent = DeepQAgent(env, display=display, params=config["params"])
    
    # Load checkpoint
    checkpoint_path = f"checkpoints/{run_name}_2000.pth"
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Run test episodes
    agent.train(config["params"], run_name, save_graphs=False)
    agent.save_results(run_name, num_episodes, False)

if __name__ == "__main__":
    env = input("Which environment do you want to run? (1 - simple / 2 - deep_q [default])\n")
    if not env:
        env = "2"

    display_input = input("Do you want to display the environment? (y / n [default])\n")
    if not display_input:
        display_input = "n"

    display = True if display_input == "y" else False

    if env == "1":
        ## Simple Q-Learning
        mars_environment = Mars_Environment(
            config = configs["simple_game"],
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
            display=display
        )

        mars_environment.run()

    elif env == "2":
        test = input("Do you want to test the best model (prioritised_double_dqn)? (y / n [default])\n")
        if not test:
            test = "n"

        test = True if test == "y" else False

        ## Deep Q-Learning
        config = configs["deep_q_game"]

        if test:
            # Testing phase
            test_config = config.copy()
            test_config["params"]["num_episodes"] = 500
            test_config["params"]["epsilon_start"] = 0.05
            test_config["params"]["save_checkpoint"] = False
            test_config["params"]["double_dqn"] = True
            test_config["params"]["prioritised"] = True

            test_model(test_config, "prioritised_double_dqn", display)
        else:
            training_model = input("Which model do you want to train? (1 - baseline / 2 - double_dqn / 3 - prioritised_double_dqn [default])\n")
            if not training_model:
                print("Defaulting to prioritised_double_dqn")
                training_model = "3"

            if training_model not in ["1", "2", "3"]:
                print("Invalid input. Please enter 1, 2, or 3.")
                exit()

            training_model = int(training_model)

            if training_model == 1:
                config["params"]["double_dqn"] = False
                config["params"]["prioritised"] = False
                train_model(config, "baseline", display)

            elif training_model == 2:
                config["params"]["double_dqn"] = True
                config["params"]["prioritised"] = False
                train_model(config, "double_dqn", display)
            
            elif training_model == 3:
                config["params"]["double_dqn"] = True
                config["params"]["prioritised"] = True
                train_model(config, "prioritised_double_dqn", display)