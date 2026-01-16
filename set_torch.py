# TODO give credit

from typing import Tuple
from strategies.base_strategy import BaseSETStrategy
from strategies.random_set import RandomSET
import torch
import torch.nn as nn
import numpy as np
from srelu_torch import SReLU
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PerPixelNormalize:

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class Dense(nn.Module):

    def __init__(self, input, out, dropout=0.3) -> None:
        super().__init__()

        self.layer = nn.Linear(input, out)
        self.activation = SReLU(out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return self.dropout(x)


class Head(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.layer = nn.Linear(4000, 10)
        # NOTE: No softmax here because CrossEntropyLoss already does softmax in torch
        # self.activation = nn.Softmax()

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Sequential(nn.Flatten())
        self.sparse1 = Dense(32 * 32 * 3, 4000)
        self.sparse2 = Dense(4000, 1000)
        self.sparse3 = Dense(1000, 4000)
        self.classifier = Head()

    def forward(self, x):
        x = self.flatten(x)
        x = self.sparse1(x)
        x = self.sparse2(x)
        x = self.sparse3(x)
        return self.classifier(x)


def create_weights_mask(epsilon, no_rows, no_cols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = torch.rand(no_rows, no_cols)
    # mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (no_rows + no_cols)) / (
        no_rows * no_cols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    parameter_count = torch.sum(mask_weights)
    print("Create Sparse Matrix: No parameters, NoRows, NoCols ",
          parameter_count, no_rows, no_cols)
    return [parameter_count, mask_weights]


class SET_MLP_CIFAR10():

    wSRelu1: torch.Tensor

    def __init__(self, strategy: BaseSETStrategy, max_epochs=1000) -> None:

        self.strategy = strategy
        self.model = MLP()

        # set model parameters
        self.epsilon = 20  # control the sparsity level as discussed in the paper
        self.zeta = 0.3  # the fraction of the weights removed
        self.batch_size = 100  # batch size
        self.max_epoches = max_epochs  # number of epochs
        self.learning_rate = 0.01  # SGD learning rate
        self.num_classes = 10  # number of classes
        self.momentum = 0.9  # SGD momentum

        self.device = 'cuda' if torch.cuda.is_available(
        ) else 'mps' if torch.mps.is_available() else 'cpu'

        # generate an Erdos Renyi sparse weights mask for each layer
        [self.parameter_count1,
         self.weight_mask1] = create_weights_mask(self.epsilon, 4000,
                                                  32 * 32 * 3)
        [self.parameter_count2,
         self.weight_mask2] = create_weights_mask(self.epsilon, 1000, 4000)
        [self.parameter_count3,
         self.weight_mask3] = create_weights_mask(self.epsilon, 4000, 1000)

        # initialize layers weights
        self.w1: torch.Tensor = self.model.sparse1.layer.weight
        self.w2: torch.Tensor = self.model.sparse2.layer.weight
        self.w3: torch.Tensor = self.model.sparse3.layer.weight
        self.w4: torch.Tensor = self.model.classifier.layer.weight

        # initialize weights for SReLu activation function
        self.wSRelu1: torch.Tensor = self.model.sparse1.activation.weight
        self.wSRelu2: torch.Tensor = self.model.sparse2.activation.weight
        self.wSRelu3: torch.Tensor = self.model.sparse3.activation.weight

        with torch.no_grad():
            self.model.sparse1.layer.weight *= self.weight_mask1
            self.model.sparse2.layer.weight *= self.weight_mask2
            self.model.sparse3.layer.weight *= self.weight_mask3

    def rewire_mask(self,
                    weights: np.ndarray,
                    noWeights,
                    mask: np.ndarray,
                    extra_info=None) -> Tuple[torch.Tensor, torch.Tensor]:

        # remove zeta largest negative and smallest positive weights
        keep_mask: np.ndarray = self.strategy.prune_neurons(weights, mask)
        rewired_mask: np.ndarray = keep_mask.reshape(
            weights.shape).astype(float)
        pruned_original_mask: np.ndarray = rewired_mask.copy()

        occupied = set(zip(*np.where(rewired_mask == 1)))
        noRewires = int(noWeights - np.sum(rewired_mask))
        new_positions = self.strategy.regrow_neurons(noRewires, weights.shape,
                                                     occupied, extra_info)

        rewired_mask[new_positions[:, 0], new_positions[:, 1]] = 1

        return (torch.from_numpy(rewired_mask).float().to(self.device),
                torch.from_numpy(pruned_original_mask).float().to(self.device))

    def weights_evolution(self):
        self.w1 = self.model.sparse1.layer.weight
        self.w2 = self.model.sparse2.layer.weight
        self.w3 = self.model.sparse3.layer.weight
        self.w4 = self.model.classifier.layer.weight

        self.wSRelu1 = self.model.sparse1.activation.weight
        self.wSRelu2 = self.model.sparse2.activation.weight
        self.wSRelu3 = self.model.sparse3.activation.weight

        for i in range(1, 4):
            weight_mask: torch.Tensor = getattr(self, f'weight_mask{i}')
            w: torch.Tensor = getattr(self, f'w{i}')
            param_count = getattr(self, f'parameter_count{i}')

            w_cpu = w.detach().cpu().numpy().flatten()
            mask_cpu = weight_mask.detach().cpu().numpy().flatten()

            match self.strategy.__class__.__name__:
                case "RandomSET":
                    new_mask, core = self.rewire_mask(w_cpu, param_count,
                                                      mask_cpu)
                # case "NeuronCentrality":
                #     new_mask, core = self.rewire_mask(w_cpu, param_count, mask_cpu, {})

                case _:
                    raise NotImplementedError(
                        f"Strategy {self.strategy.__class__.__name__} not implemented"
                    )

            with torch.no_grad():
                setattr(self, f'weight_mask{i}',
                        new_mask.reshape(weight_mask.shape))
                w *= core.reshape(w.shape)
                setattr(self, f'w{i}', w)

        # match self.strategy.__class__.__name__:
        #     case "RandomSET":
        #         [self.weight_mask1,
        #          wm1Core] = self.rewire_mask(self.w1, self.parameter_count1,
        #                                      self.weight_mask1)
        #         [self.weight_mask2,
        #          wm2Core] = self.rewire_mask(self.w2, self.parameter_count2,
        #                                      self.weight_mask2)
        #         [self.weight_mask3,
        #          wm3Core] = self.rewire_mask(self.w3, self.parameter_count3,
        #                                      self.weight_mask3)
        #     case "NeuronCentrality":
        #         [wm1, wm1Core] = self.rewire_mask(w1, self.mask1, self.wm1, {
        #             "layer": "layer_1",
        #             "self": self
        #         })
        #         [wm2, wm2Core] = self.rewire_mask(w2, self.mask2, self.wm2, {
        #             "layer": "layer_2",
        #             "self": self
        #         })
        #         [wm3, wm3Core] = self.rewire_mask(w3, self.mask3, self.wm3, {
        #             "layer": "layer_3",
        #             "self": self
        #         })
        #     case _:
        #         raise NotImplementedError(
        #             f"Strategy {self.strategy.__class__.__name__} not implemented"
        #         )

        # self.w1 *= wm1Core
        # self.w2 *= wm2Core
        # self.w3 *= wm3Core

        # with torch.no_grad():
        #     self.model.sparse1.layer.weight = self.w1
        #     self.model.sparse2.layer.weight = self.w2
        #     self.model.sparse3.layer.weight = self.w3
        #     self.model.classifier.layer.weight = self.w4
        #
        #     self.model.sparse1.activation.weight = self.wSRelu1
        #     self.model.sparse2.activation.weight = self.wSRelu2
        #     self.model.sparse3.activation.weight = self.wSRelu3

    def _setup_training_data(self):
        temp_dataset = datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
        temp_loader = DataLoader(temp_dataset, batch_size=len(temp_dataset))
        all_data = next(iter(temp_loader))[0]  # Shape: (50000, 3, 32, 32)

        # per-pixel mean and std (matching Keras axis=0)
        pixel_mean = all_data.mean(dim=0)  # Shape: (3, 32, 32)
        pixel_std = all_data.std(dim=0)  # Shape: (3, 32, 32)

        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            PerPixelNormalize(pixel_mean, pixel_std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             PerPixelNormalize(pixel_mean, pixel_std)])

        train_dataset = datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=train_transform)

        test_dataset = datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=test_transform)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=2)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=2)

        return train_loader, test_loader

    def _eval(self, dataloader):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        return accuracy

    def train(self, target_accuracy=1.0):
        [train_loader, test_loader] = self._setup_training_data()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum)

        best_acc = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'epoch_count': 0,
        }

        self.model.to(self.device)

        print(
            f"starting training for max {self.max_epoches} epochs ({target_accuracy=})"
        )

        for epoch in range(self.max_epoches):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            batch_num = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += images.size(0)
                train_correct += predicted.eq(labels).sum().item()
                batch_num += 1

            train_acc = train_correct / train_total

            test_acc = self._eval(test_loader)

            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['train_loss'].append(train_loss)

            if test_acc > best_acc:
                best_acc = test_acc

            print(
                f'Epoch {epoch+1:3d}/{self.max_epoches} | Loss: {train_loss:.3f} | Acc: {test_acc:.2f} | Best: {best_acc:.2f}'
            )

            if test_acc >= target_accuracy:
                print(f"reached target accuracy ({target_accuracy})!")
                history['epoch_count'] = epoch
                break

            self.weights_evolution()

        return history


if __name__ == '__main__':
    set_strategy = RandomSET()

    # create and run a SET-MLP model on CIFAR10
    model = SET_MLP_CIFAR10(set_strategy, max_epochs=60)

    # train the SET-MLP model until 40%
    history = model.train(target_accuracy=0.4)
    print(f"took {history['epoch_count']} epochs until convergance")

    # save accuracies over for all training epochs
    # in "results" folder you can find the output of running this file
    np.savetxt("results/set_mlp_srelu_sgd_cifar10_acc.txt",
               np.asarray(history['test_acc']))
