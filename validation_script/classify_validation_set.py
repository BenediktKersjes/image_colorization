# adapted from https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py

import os
from collections import defaultdict

import torch
from PIL import Image

from torchvision import models, transforms

from config import images_path

if __name__ == '__main__':

    subfolder = '128_2018_09_23__05_08_51_DeepKoalarizationNorm_MultinomialCrossEntropyLoss_MITPlaces_iter31000'
    grayscale = False

    # architecture to use
    arch = 'alexnet'

    # load the pre-trained weights
    model_file = './data/%s_places365.pth.tar' % arch
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    centre_crop = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load category names
    file_name = './data/categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # load correct labels
    label_dict = defaultdict()
    file_name = './data/val.txt'
    with open(file_name) as class_file:
        for line in class_file:
            label_dict[line.strip().split(' ')[1]] = line.strip().split(' ')[0]

    top1 = 0
    top5 = 0
    path = images_path + 'validation/' + subfolder + '/'
    for index, file in enumerate(os.listdir(path)):
        if index % 100 == 0:
            print(index)

        if file == 'images.txt':
            continue

        img = Image.open(path + file)

        if grayscale:
            img = transforms.Grayscale(num_output_channels=3)(img)

        input_img = centre_crop(img).unsqueeze(0)

        logit = model.forward(input_img)
        h_x = torch.nn.functional.softmax(logit, 1).squeeze()
        probs, idx = h_x.sort(0, True)

        # print('{} prediction on {}'.format(arch, file))
        for i in range(0, 5):
            # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

            if classes[idx[i]] == label_dict[file]:
                if i == 1:
                    top1 += 1

                top5 += 1

    print('top1 accuracy: {}'.format((top1 / 1200.)))
    print('top5 accuracy: {}'.format((top5 / 1200.)))
