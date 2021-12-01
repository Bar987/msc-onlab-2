from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common import logger
from datetime import datetime
import os

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.tb_formatter = None
        self.actions = []
        self.rewards = []
        self.iter = 0

    def _on_step(self) -> bool:
        if self.tb_formatter:
            self.actions.append(self.training_env.get_attr('last_action'))
            self.rewards.append(self.training_env.get_attr('last_reward'))
        else:
            self.tb_formatter = TensorBoardOutputFormat(logger.get_dir())
        return True

    def _on_rollout_end(self) -> None:
        self.tb_formatter.writer.add_histogram("rewards", np.array(self.rewards), self.iter)
        self.tb_formatter.writer.add_histogram("actions", np.array(self.actions), self.iter)
        self.iter += 1

class CheckpointCallback(BaseCallback):
    
    def __init__(self, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.rewards = []
        self.max_reward = -9999999
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> None:
        curr_rewards = np.array(self.training_env.get_attr('last_reward'))
        self.rewards.append(curr_rewards)
                                
    def _on_rollout_end(self) -> None:
        temp = np.stack(self.rewards)
        sum = np.sum(temp)
        if sum > self.max_reward:
            self.max_reward = sum
            self.rewards = []
            time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            path = os.path.join(self.save_path, f"{self.name_prefix}_{time}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")