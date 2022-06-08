# Import the modules
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from callback import Callback
from model import KerasModel
from setmario import env
from train import declareParserArguments


from tensorflow.python.client import device_lib

"""tf.config.set_visible_devices([], "GPU")
print(device_lib.list_local_devices())"""

if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="Ai for SuperMario")
    args = declareParserArguments(parser=parser)

    # Create a callback object(Setup model saving callback)
    callback = Callback(check_freq=10000, args=args)

    while True:  # Run until solved
        state = env.reset()
        episode_reward = 0
        eps = np.finfo(
            np.float32
        ).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        keras_obj = KerasModel(env=env, callback=callback, args=args)
        with tf.GradientTape() as tape:
            for timestep in range(1, keras_obj.max_steps_per_episode):
                env.render()

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = keras_obj._keras_model(state=state)
                keras_obj.critic_value_history.append(critic_value(0, 0))

                # Sample action from action probability distribution
                action = np.random.choice(
                    keras_obj.num_actions, p=np.squeeze(action_probs)
                )
                keras_obj.action_probs_history.append(
                    tf.math.log(action_probs[0, action])
                )

                # Apply the sampled action in our environment
                state, reward, done, info = env.step(action)
                keras_obj.rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break

            # Update running reward to check condition for solving
            keras_obj.running_reward = (
                0.05 * episode_reward + (1 - 0.05) * keras_obj.running_reward
            )
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in keras_obj.rewards_history[::-1]:
                discounted_sum = r + keras_obj.gamma * discounted_sum
                returns.insert(0, discounted_sum)
            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            # Calculating loss values to update our network
            history = zip(
                keras_obj.action_probs_history,
                keras_obj.critic_value_history,
                returns,
            )
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss
                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    keras_obj.loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )
            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, keras_obj.model.trainable_variables)
            keras_obj.optimizer.apply_gradients(
                zip(grads, keras_obj.model.trainable_variables)
            )
            # Clear the loss and reward history
            keras_obj.action_probs_history.clear()
            keras_obj.critic_value_history.clear()
            keras_obj.rewards_history.clear()

        # Log details
        keras_obj.episode_count += 1
        if keras_obj.episode_count % 10 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(keras_obj.running_reward, keras_obj.episode_count))

        if keras_obj.running_reward > 195:  # Condition to consider the task solved
            print("Solved at episode {}!".format(keras_obj.episode_count))
            break
