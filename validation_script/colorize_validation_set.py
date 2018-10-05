import os

import torch
from scipy.misc import imsave
from torch.utils.data import DataLoader

from config import images_path, trained_models_path
from image_generator import generate_images_numpy
from network import DeepKoalarization, ColorfulImageColorization, DeepKoalarizationNorm
from colorization_dataset import ColorizationDataset


def colorize(network, load_model, image_size, regression):
    if regression:
        out_channels = 2
    else:
        out_channels = 256

    assert network in ['DeepKoalarization', 'ColorfulImageColorization', 'DeepKoalarizationNorm']

    if network == 'DeepKoalarization':
        model = DeepKoalarization(out_channels=out_channels, to_rgb=False)
    elif network == 'DeepKoalarizationNorm':
        model = DeepKoalarizationNorm(out_channels=out_channels, to_rgb=False)
    else:
        model = ColorfulImageColorization(out_channels=out_channels, to_rgb=False)

    state_dict = torch.load(trained_models_path + load_model + '.pth', map_location='cpu')
    model.load_state_dict(state_dict['model'])

    validation_loader = DataLoader(
        ColorizationDataset(
            images_path + 'validation/',
            train=False,
            size=(image_size, image_size),
            target_rgb=False,
            convert_to_categorical=False,
            do_not_convert=False,
            validation=True),
        batch_size=10)

    if not os.path.exists(images_path + 'validation/' + str(image_size) + '_' + load_model):
        os.makedirs(images_path + 'validation/' + str(image_size) + '_' + load_model)

    for data, file in validation_loader:
        if torch.cuda.is_available():
            data = data.cuda()

        output = model(data)
        images = generate_images_numpy(data, output, False, regression=regression, t=.38)

        for idx, image_array in enumerate(images):
            imsave(images_path + 'validation/' + str(image_size) + '_' + load_model + '/' + file[idx], image_array)


if __name__ == '__main__':
    colorize('ColorfulImageColorization', '2018_09_21__16_58_21_ColorfulImageColorization_MSELoss_MITPlaces_iter25600', 128, True)
    colorize('ColorfulImageColorization', '2018_09_21__16_58_21_ColorfulImageColorization_MSELoss_MITPlaces_iter25600', 256, True)

    colorize('ColorfulImageColorization', '2018_09_20__16_03_43_ColorfulImageColorization_MultinomialCrossEntropyLoss_MITPlaces_iter19000', 128, False)
    colorize('ColorfulImageColorization', '2018_09_20__16_03_43_ColorfulImageColorization_MultinomialCrossEntropyLoss_MITPlaces_iter19000', 256, False)

    colorize('DeepKoalarizationNorm', '2018_09_22__20_37_42_DeepKoalarizationNorm_MSELoss_MITPlaces_iter26800', 128, True)
    colorize('DeepKoalarizationNorm', '2018_09_22__20_37_42_DeepKoalarizationNorm_MSELoss_MITPlaces_iter26800', 256, True)

    colorize('DeepKoalarizationNorm', '2018_09_23__05_08_51_DeepKoalarizationNorm_MultinomialCrossEntropyLoss_MITPlaces_iter31000', 128, False)
    colorize('DeepKoalarizationNorm', '2018_09_23__05_08_51_DeepKoalarizationNorm_MultinomialCrossEntropyLoss_MITPlaces_iter31000', 256, False)
