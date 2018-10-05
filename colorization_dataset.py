from __future__ import print_function

import os
import os.path

import numpy as np
import pyprind
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

import pytorch_differential_color as pdc
from config import grid_size


def conversion_image(ab_img):
    # fast conversion to
    img_size = ab_img.shape[-2:]
    ab_img = (ab_img * 100 + 128).float()
    grid_shape = (int(256 / grid_size), int(256 / grid_size))
    # a_img of shape [size[0], size[1], grid[0], grid[1]]
    a_img, b_img = ab_img.expand(*grid_shape, *ab_img.shape).permute(2, 3, 4, 0, 1)
    grid_a, grid_b = torch.meshgrid([torch.arange(0, 255, grid_size).float(),
                                     torch.arange(0, 255, grid_size).float()])
    diff_a = torch.clamp(torch.abs(a_img - grid_a) / grid_size, 0, 1)
    diff_b = torch.clamp(torch.abs(b_img - grid_b) / grid_size, 0, 1)
    weight_ab = (1 - diff_a) * (1 - diff_b)
    new_shape = (*img_size, -1)
    target = weight_ab.view(*new_shape).permute(2, 0, 1)

    return target


def conversion_batch(ab_batch):
    batch_size = ab_batch.shape[0]
    img_size = ab_batch.shape[-2:]
    ab_batch = (ab_batch * 100 + 128).float()
    grid_shape = (int(256 / grid_size), int(256 / grid_size))
    # a_img of shape [batch_size, size[0], size[1], grid[0], grid[1]]
    a_img, b_img = ab_batch.expand(*grid_shape, *ab_batch.shape).permute(3, 2, 4, 5, 0, 1)
    my_range = torch.arange(0, 255, grid_size).float()
    if ab_batch.is_cuda:
        my_range = my_range.cuda()
    del ab_batch
    grid_a, grid_b = torch.meshgrid([my_range,
                                     my_range])
    del my_range
    diff_a = torch.clamp(torch.abs(a_img - grid_a) / grid_size, 0, 1)
    del a_img, grid_a
    diff_b = torch.clamp(torch.abs(b_img - grid_b) / grid_size, 0, 1)
    del b_img, grid_b
    weight_ab = (1 - diff_a) * (1 - diff_b)
    new_shape = (batch_size, *img_size, -1)
    target = weight_ab.view(*new_shape).permute(0, 3, 1, 2)

    return target


class ColorizationDataset(data.Dataset):
    """
    just reads all files from root subfolder. In root folder there has to be one folder 'train' and one 'test'
    """

    def __init__(self, folder, train=True, size=None, convert_to_categorical=True, use_own_converter=True,
                 target_rgb=False, save_data=False, do_not_convert=False, validation=False):
        self.folder = folder + 'train/' if train else folder + 'test/'
        self.images = self.load_images_list()
        self.size = (224, 224) if size is None else size
        self.convert_to_categorical = convert_to_categorical
        self.use_own_converter = use_own_converter
        self.target_rgb = target_rgb
        self.save_data = save_data
        self.do_not_convert = do_not_convert
        self.validation = validation

    def convert_all_slow(self):
        progress = pyprind.ProgBar(len(self.images), update_interval=.5, title='Conversions')
        for index in range(len(self.images)):
            self.__getitem__(index)
            progress.update()
        print(progress)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = Image.open(self.folder + self.images[index])

        # resize and crop
        img = transforms.Resize(min(self.size))(img)
        img = transforms.CenterCrop(self.size)(img)

        # convert
        rgb = img.convert("RGB")

        if self.do_not_convert:
            input_data = torch.from_numpy(np.array(rgb)).permute(2, 0, 1)
            target = input_data
        else:
            rgb_tensor = to_tensor(rgb)
            lab = pdc.rgb2lab(rgb_tensor)

            # select data for grey and target
            input_data, ab = torch.split(lab, [1, 2], dim=0)
            if self.target_rgb:
                target = rgb_tensor
            else:
                if self.convert_to_categorical:
                    # fast conversion
                    target = conversion_image(ab)
                else:
                    target = ab

            if self.save_data:
                torch.save(input_data, self.folder + self.images[index] + '.grey.pth')
                torch.save(target, self.folder + self.images[index] + '.color.pth')

        if self.validation:
            return input_data, self.images[index]
        else:
            return input_data, target

    def load_images_list(self):
        index_path = self.folder + 'images.txt'
        if os.path.isfile(index_path):
            with open(index_path, 'r') as f:
                images = f.read().splitlines()
        else:
            images = os.listdir(self.folder)
            with open(index_path, 'w') as f:
                f.writelines('{}\n'.format(image) for image in images)
        return images


if __name__ == "__main__":
    my_dataset = ColorizationDataset(
        './data/images/',
        train=True,
        size=(64, 64),
        target_rgb=False)

    my_dataset.convert_all_slow()
