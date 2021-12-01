
from puzzle_gym.envs.puzzle_env import PuzzleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
import statistics
import wandb
import torch.nn as nn
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from callbacks import CheckpointCallback, TensorboardCallback
import os
from custom_cnn import CustomCNN

OBS_CONF = {"min": 0, "max": 1, "type": np.float32}
CHANNEL_NUM = 3
IMG_SIZE = (87, 87)
FEATURES_DIM = 256
LEARNING_RATE = 0.0005
MODEL_TYPE = "cifar"
NET_ARCH = "siamese"
PARAMS = {
    'images': "",
    'img_size': IMG_SIZE,
    'max_step_num': 100,
    'puzzle_size': (3, 3),
    'puzzle_type': "switch",
    'dist_type': "manhattan",
    'penalty_for_step': -0.1,
    'reward_for_completiton': 25,
    'obs_type': "split"
}


def load_data(source):

    if source == "dataset":

        stats = ((0.448109, 0.43899548, 0.40594447),
                 (0.22454526, 0.22170316, 0.22334038))

        transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomCrop(87),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*stats, inplace=True)])

        valid_tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*stats)])

        train_dataset = datasets.STL10(
            './STL10', split='train+unlabeled', transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.STL10(
            './STL10', split='test', transform=transforms.ToTensor(), download=True)

        train_dataset.transform = transform
        test_dataset.transform = valid_tfms

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        train_dataset_array = np.moveaxis(
            next(iter(train_loader))[0].numpy(), 1, -1)
        test_dataset_array = np.moveaxis(
            next(iter(test_loader))[0].numpy(), 1, -1)

    return train_dataset_array, test_dataset_array


def train():

    train_images, test_images = load_data("dataset")

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=FEATURES_DIM, model_type=MODEL_TYPE, net_arch=NET_ARCH),
        normalize_images=False,
        activation_fn=nn.ReLU,
    )

    train_params = PARAMS
    train_params['images'] = train_images

    config = {
        "dataset_size": train_images.shape[0],
        "model": MODEL_TYPE,
        "net_arch": NET_ARCH,
        "features_dim": FEATURES_DIM,
        "learning_rate": LEARNING_RATE,
        "img_size": PARAMS["img_size"],
        "puzzle_size": PARAMS["puzzle_size"],
        "max_step_num": PARAMS["max_step_num"],
        "puzzle_type": PARAMS["puzzle_type"],
        "dist_type": PARAMS["dist_type"],
        "penalty_for_step": PARAMS["penalty_for_step"],
        "reward_for_completiton": PARAMS["reward_for_completiton"],
    }

    run = wandb.init(project="msc-onlab-2", entity="osvathb",
                     sync_tensorboard=True, config=config)

    env = make_vec_env(PuzzleEnv, n_envs=10, seed=42,
                       monitor_dir='/home1/rlpuzzle/monitor', env_kwargs=train_params)

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=LEARNING_RATE,
                max_grad_norm=0.8, seed=42, tensorboard_log=f"runs/{run.id}")

    checkpoint_callback = CheckpointCallback(
        save_path='./models/', name_prefix=MODEL_TYPE+'_'+NET_ARCH)
    tensorboard_callback = TensorboardCallback()
    wandb_callback = WandbCallback(verbose=2, gradient_save_freq=1000)

    callback = CallbackList([checkpoint_callback, wandb_callback, tensorboard_callback
                             ])

    model.learn(total_timesteps=5000000, callback=callback)

    test(model, test_images)


def test(model, test_images):
    test_params = PARAMS
    test_params['images'] = test_images
    test_env = Monitor(PuzzleEnv, n_envs=10, seed=42,
                       monitor_dir='/home1/rlpuzzle/monitor', env_kwargs=test_params)

    solutions = []
    rews = []
    steps = []
    sample = len(test_images)
    errors = 0

    for iter in range(sample):
        i = 0
        done = False
        obs = test_env.reset()
        frames = [obs]

        while not done:
            i += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)
            frames.append(obs)
            rews.append(rewards)

            if i == 10000:
                errors += 1
                break

        solutions.append(frames)
        done = False
        print(i, sum(rews), rews)
        rews = []
        steps.append(i)

    print('Average steps taken:  ', sum(steps) / sample)
    print('Median of steps taken: ', statistics.median(steps))
    print('Number of errors: ', errors)
    plt.hist(steps, bins=9)
    plt.savefig('fig.png')


def clone_used_repo():
    os.system('git clone https://github.com/akamaster/pytorch_resnet_cifar10.git')
    os.system('echo > /pytorch_resnet_cifar10/__init__.py')


if __name__ == "__main__":
    clone_used_repo()
    train()
