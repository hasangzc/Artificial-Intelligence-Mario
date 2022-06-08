# import the PPO
from stable_baselines3 import PPO

# Import the Arguments
from argparse import ArgumentParser

# Import the modules for model2
import tensorflow as tf
from tensorflow import keras
from keras import layers


class ReinforcementModel:
    def __init__(self, env, callback, args: ArgumentParser):
        self.env = env
        self.callback = callback
        self._args = args

    def _train_model(self):
        self.model = PPO(
            "CnnPolicy",  # When create a reinforcement learning behind the scenes we have policy network
            # Cnn: Spesific type of neural network
            self.env,
            verbose=1,  # Get a whole bunch of information when you start training
            tensorboard_log=self._args.logs_dir,
            learning_rate=0.000001,
            n_steps=512,  # How many frames we are going to wait per game before we go and update our neural network
        )

        self.model.learn(
            total_timesteps=1000000,
            callback=self.callback,
        )

    def _load_model(self):
        # Load Model
        self.model = PPO.load(f"{self._args.checkpoint_dir}/{self._args.saved_model}")
        return self.model


class KerasModel:
    def __init__(self, env, callback, args: ArgumentParser):
        self.gamma = 0.99
        self.max_steps_per_episode = 10000

        self.env = env
        self.callback = callback
        self._args = args

        self.num_inputs = 7
        self.num_actions = 7
        self.num_hidden = 128

        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.episode_count = 0

    def _keras_model(self):
        self.inputs = layers.Input(shape=(self.num_inputs,))
        self.common = layers.Dense(
            self.num_hidden,
            activation="relu",
        )(self.inputs)
        self.action = layers.Dense(
            self.num_actions,
            activation="softmax",
        )(self.common)
        self.critic = layers.Dense(1)(self.common)

        self.model = keras.Model(inputs=self.inputs, outputs=[self.action, self.critic])
