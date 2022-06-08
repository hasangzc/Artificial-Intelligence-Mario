# Import game
import gym_super_mario_bros

# Import the Frame Stacker Wrapper and GrayScalig Wrapper
from gym.wrappers import GrayScaleObservation

# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Import Joypad
from nes_py.wrappers import JoypadSpace

# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Import visualisation lib
import matplotlib.pyplot as plt

# Set up Mario Game
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # 7 Action

# Preprocess Env
# GrayScale
env = GrayScaleObservation(env=env, keep_dim=True)
# Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])  # Return env back
# Stack the frames
env = VecFrameStack(
    env, 4, channels_order="last"
)  # 4: How many frames we want to stack  channels_order:


done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step([4])
    env.render()

env.close()
