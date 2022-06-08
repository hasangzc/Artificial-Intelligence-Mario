from argparse import ArgumentParser

# Import the model
from model import ReinforcementModel

# Import the callback
from callback import Callback

# Import the env variable
from setmario import env


def declareParserArguments(parser: ArgumentParser) -> ArgumentParser:
    # Add arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./train/",
        # required=True,
        help="Saved models directory",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="./logs/",
        # required=True,
        help="Every time we run the algorithm once we gonna create log file",
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        default="best_model_20000",
        # required=True,
        help="Saved model",
    )

    # Return the parsed arguments
    return parser.parse_args()


# If this script is run directly
if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="Ai for SuperMario")
    args = declareParserArguments(parser=parser)

    # Create a callback object(Setup model saving callback)
    callback = Callback(check_freq=10000, args=args)
    # Create a trainer object
    trainer = ReinforcementModel(env=env, callback=callback, args=args)

    # Train the model
    trainer._train_model()
