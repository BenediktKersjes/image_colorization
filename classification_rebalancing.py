import math
import torch
from torch import nn
import pyprind

from config import images_path, grid_size
from colorization_dataset import ColorizationDataset, conversion_batch


class GaussianFilter(nn.Module):
    # adapted
    # from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
    def __init__(self, channels, sigma, kernel_size):
        super(GaussianFilter, self).__init__()

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        nom = -torch.sum((xy_grid - mean) ** 2., dim=-1).float()
        denom = (2 * variance)
        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(nom.float() / denom)
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                         kernel_size=kernel_size, groups=channels, bias=False,
                                         padding=(kernel_size - 1) // 2)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, input):
        return self.gaussian_filter(input)


def calc_rebalancing(sigma=5, weighting_factor=0.5, image_size=96):
    # init variables
    bins_per_channel = int(256 / grid_size)
    bins = bins_per_channel ** 2
    dataloader = torch.utils.data.DataLoader(
        ColorizationDataset(
            images_path,
            train=True,
            size=(image_size, image_size),
            convert_to_categorical=False),
        batch_size=64,
        shuffle=False)
    class_count = torch.zeros(bins_per_channel ** 2)
    gaussian_filter = GaussianFilter(channels=1, sigma=5 / grid_size, kernel_size=2 * int(sigma / grid_size + 1) + 1)
    # loop through data and count classes
    progress = pyprind.ProgBar(len(dataloader), update_interval=.5, title='Conversions')
    for data, target in dataloader:
        q_star = conversion_batch(target.cuda()).argmax(dim=1)
        class_count += q_star.cpu().float().histc(bins=bins, min=0, max=bins - 1)
        progress.update()
    print(progress)
    # make again a and b dimension separate and make to distribution
    distribution = class_count.float() / class_count.sum()
    # filter distribution with gaussian kernel
    distribution_filtered = gaussian_filter(distribution.view(1, 1, bins_per_channel, bins_per_channel)).view(-1)
    # calc weights
    w = 1 / ((1 - weighting_factor) * distribution_filtered + weighting_factor * 1 / bins)
    # normalize weights
    w /= (w * distribution_filtered).sum()

    return w, (class_count, distribution, distribution_filtered)


if __name__ == "__main__":
    with torch.no_grad():
        weighting_factor = 0.5
        # calc weights
        w, (class_count, distribution, distribution_filtered) = calc_rebalancing(weighting_factor=weighting_factor)
        # save weights
        torch.save(w, images_path + 'classification_weights_{}_{}.pth'.format(grid_size, str(weighting_factor)))
        torch.save(class_count, images_path + 'class_counts_{}.pth'.format(grid_size))
        # show weights
        bins_per_channel = int(256 / grid_size)
        import matplotlib.pyplot as plt

        for data in [w]:
            plt.imshow(data.view(bins_per_channel, bins_per_channel).numpy())
            plt.show()
        for data in [class_count, distribution, distribution_filtered, distribution_filtered - distribution]:
            plt.imshow(data.log().view(bins_per_channel, bins_per_channel).numpy())
            plt.show()
