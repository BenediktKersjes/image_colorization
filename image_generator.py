import torch
import numpy as np
from torch.utils.data import DataLoader
from config import grid_size

import pytorch_differential_color as pdc


def generate_images(luminance, chrominance, is_target, t=0.38, regression=False):
    if regression is False:
        # simulated annaling, like in Colorful Image Colorization
        if is_target:
            z = chrominance
        else:
            z = torch.nn.Softmax2d()(chrominance)
        chrominance_temp_corrected_permuted = torch.nn.Softmax2d()(torch.log(z) / t).permute(0, 2, 3, 1)
        my_range = torch.arange(-128, 127, grid_size).float() / 100.
        if chrominance.is_cuda:
            my_range = my_range.cuda()
        grid_a, grid_b = torch.meshgrid([my_range, my_range])
        
        a = torch.sum(chrominance_temp_corrected_permuted * grid_a.contiguous().view(-1), dim=3).unsqueeze(1)
        b = torch.sum(chrominance_temp_corrected_permuted * grid_b.contiguous().view(-1), dim=3).unsqueeze(1)

        imgs = torch.cat((luminance, a, b), 1)
    else:
        imgs = torch.cat((luminance, chrominance), 1)

    return pdc.lab2rgb(imgs)


def generate_images_numpy(luminance, chrominance, is_target, t=0.38, regression=False):
    return generate_images(luminance.detach().cpu(), chrominance.detach().cpu(), is_target, t, regression)\
        .permute(0, 2, 3, 1).numpy().astype(np.float64)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from config import *
    from colorization_dataset import ColorizationDataset
    from network import DeepKoalarization
    print('finished importing')

    batch_size = 1

    test_loader = DataLoader(
        ColorizationDataset(images_path, train=False),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = iter(test_loader)
    print('initialized test loader')

    data, target = next(test_loader)    
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()
    print('got image')

    images = generate_images_numpy(data, target, is_target=True)
    print('generated target image')
    
    for image in images:
        plt.imshow(image)
        plt.show()

    model = DeepKoalarization(out_channels=1024)
    if torch.cuda.is_available():
        model.cuda()

    output = model(data)

    images = generate_images_numpy(data, output, is_target=False)
    print('generated output image')

    for image in images:
        plt.imshow(image)
        plt.show()
