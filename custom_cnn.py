import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import timm
from pytorch_resnet_cifar10 import resnet


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, model_type='resnet18d', net_arch="regular"):
        

        super(CustomCNN, self).__init__(observation_space, features_dim)

        self.net_arch = net_arch

        if model_type == 'custom':
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            )

            self.later = nn.Sequential(
                nn.Linear(576, features_dim),
                nn.ReLU(),)

        elif model_type == 'cifar':
            temp = torch.load(
                'pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th')['state_dict']
            weights = {}
            for key in temp.keys():
                new_key = key[7:]
                weights[new_key] = temp.get(key)

            self.cnn = resnet.resnet20()
            self.cnn.load_state_dict(weights)

            self.cnn.linear = nn.Identity()

            self.later = nn.Sequential(
                nn.Linear(576, features_dim),
                nn.ReLU(),)

        else:
            self.cnn = timm.create_model(model_type,
                                         pretrained=True,
                                         num_classes=0,
                                         )
            for layer in self.cnn.parameters():
                layer.requires_grad = False

            self.later = nn.Sequential(
                nn.Conv2d(512, 64, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(576, features_dim),
                nn.ReLU(),
            )

        self.isPrint = True

    def forward(self, observations):

        if self.net_arch == "siamese":
            vecs = []
            observations = torch.moveaxis(observations, 1, 0)
            for obs in observations:
                cnn_out = self.cnn(obs)
                vecs.append(cnn_out)

            features = torch.cat(vecs, 1)
            fc_out = self.later(features)

            if self.isPrint:
                print(fc_out.shape)
                self.isPrint = False

            return fc_out

        else:
            '''
            forward_features függvény után jönne csak a global pooling, ezért van ez használva
            '''
            cnn_out = self.cnn.forward_features(observations)
            fc_out = self.later(cnn_out)
            if self.isPrint:
                print(observations.shape)
                print(observations.min(), "  ", observations.max())
                print(cnn_out.shape)
                self.isPrint = False

            return fc_out