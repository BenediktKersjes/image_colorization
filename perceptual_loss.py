import torch
import torch.nn as nn
from torchvision import models
from config import grid_size


norm_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
norm_std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)


def normalize(images, norm_mean=norm_mean, norm_std=norm_std):
    if images.is_cuda:
        norm_mean = norm_mean.cuda()
        norm_std = norm_std.cuda()
    return (images - norm_mean) / norm_std


class Vgg16(nn.Module):
    # from https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


def gram(x):
    """ Calculate Gram matrix (G = FF^T) """
    # from https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_t = f.transpose(1, 2)
    g = f.bmm(f_t) / (ch * h * w)
    return g


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg16()
        self.loss_mse = nn.MSELoss()

    def forward(self, actual, expected):
        # adapted from https://github.com/dxyang/StyleTransfer/blob/master/style.py
        # get vgg features
        input_features = self.vgg(normalize(actual))
        target_features = self.vgg(normalize(expected))

        # calculate style loss
        input_gram = [gram(fmap) for fmap in input_features]
        target_gram = [gram(fmap) for fmap in target_features]
        
        style_loss = 0.0
        for j in range(4):
            style_loss += self.loss_mse(input_gram[j], target_gram[j])
        
        return style_loss


if __name__ == "__main__":
    import numpy as np
    from torch import optim
    from network import DeepKoalarization
    from colorization_dataset import ColorizationDataset
    from config import images_path

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(
        ColorizationDataset(images_path,
                      train=True),
        batch_size=batch_size
    )
    model = DeepKoalarization(out_channels=int((256/grid_size)**2), to_rgb=True)
    loss_fn = PerceptualLoss()
    print(loss_fn)
    optimizer = optim.Adam(model.parameters(), eps=1e-4)

    if torch.cuda.is_available():
        model.cuda()
        loss_fn.cuda()

    for iteration, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)

        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        max_grad = np.min([grad.abs().min().item() for grad in model.parameters()])
        print(iteration, max_grad, loss.item())

        del data, target, output, loss

        if np.isnan(max_grad):
            break
