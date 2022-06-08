# Import the modules
from argparse import ArgumentParser

from callback import Callback
from model import ReinforcementModel
from setmario import env
from train import declareParserArguments
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="Ai for SuperMario")
    args = declareParserArguments(parser=parser)

    # Create a callback object(Setup model saving callback)
    callback = Callback(check_freq=10000, args=args)

    # Create a ReinforcementModel
    obj = ReinforcementModel(env=env, callback=callback, args=args)

    # Load the model
    model = obj._load_model()

    # Start the game
    state = env.reset()

    # What is In this particular env the best keys to press
    best_keys = SIMPLE_MOVEMENT[model.predict(state)[0][0]]
    # print(best_keys)

    # Loop through the game
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()
