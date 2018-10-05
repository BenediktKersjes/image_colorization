from torch import nn
import torch

from config import images_path, grid_size


class MultinomialCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super(MultinomialCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.soft_max = nn.Softmax2d()

        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, actual, expected):
        actual = self.soft_max(actual)
        v = self.weights[expected.argmax(dim=1, keepdim=True)]
        return -(v * (expected * actual.log()).sum(dim=1)).mean()


if __name__ == '__main__':
    data = nn.Softmax2d()(torch.rand(2, 256, 128, 128))
    target = nn.Softmax2d()(torch.rand(2, 256, 128, 128))

    w = torch.load(images_path + 'classification_weights_{}.pth'.format(grid_size))
    loss = MultinomialCrossEntropyLoss(weights=w)
    print(loss.forward(data, target))
