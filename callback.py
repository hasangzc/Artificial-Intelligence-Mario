# import os the for file path management
import os

# Import the arguments
from argparse import ArgumentParser

# Import Base Callback for saving models(For example save model every 10000 games or steps)
from stable_baselines3.common.callbacks import BaseCallback


class Callback(BaseCallback):
    def __init__(self, check_freq, args: ArgumentParser, verbose=1):
        super(Callback, self).__init__(verbose)
        self.check_freq = check_freq
        self._args = args

    def _init_callback(self):
        if self._args.checkpoint_dir is not None:
            os.makedirs(self._args.checkpoint_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self._args.checkpoint_dir, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)
