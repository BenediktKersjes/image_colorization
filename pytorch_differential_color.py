# author Hannes Perrot
# inspired by https://github.com/lucasb-eyer/go-colorful
# rgb formula adapted from http://www.easyrgb.com/en/math.php

import torch
from torch.nn import functional as f

# A util to convert RGB and CIE L*ab representation in pytorch differentiable.
# RGB: All three of Red, Green and Blue in [0..1].
# CIE-XYZ: CIE's standard color space, almost in [0..1].
# CIE-L*a*b*: A perceptually uniform color space,
# i.e. distances are meaningful. L* in [0..1] and a*, b* almost in [-1..1].


# RGB
# RGB: All three of Red, Green and Blue in [0..1].
def xyz2rgb(xyz):
    """
    input xyz as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    transform_tensor = torch.tensor([[3.2404542, - 1.5371385, - 0.4985314],
                                     [-0.9692660,   1.8760108,   0.0415560],
                                     [0.0556434, - 0.2040259,   1.0572252]])
    if xyz.is_cuda:
        transform_tensor = transform_tensor.cuda()
    transform_tensor.unsqueeze_(2).unsqueeze_(3)
    if len(xyz.shape) == 4:
        convolved = f.conv2d(xyz, transform_tensor)
    else:
        convolved = f.conv2d(xyz.unsqueeze(0), transform_tensor).squeeze(0)
    # return convolved
    return torch.where(convolved > 0.0031308, 1.055 * (convolved.pow(1./2.4)) - 0.055, 12.92 * convolved)


def rgb2xyz(rgb):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    rgb = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)

    transform_tensor = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                     [0.2126729, 0.7151522, 0.0721750],
                                     [0.0193339, 0.1191920, 0.9503041]])
    if rgb.is_cuda:
        transform_tensor = transform_tensor.cuda()
    transform_tensor.unsqueeze_(2).unsqueeze_(3)
    if len(rgb.shape) == 4:
        return f.conv2d(rgb, transform_tensor)
    else:
        return f.conv2d(rgb.unsqueeze(0), transform_tensor).squeeze(0)


# LAB
# CIE-L*a*b*: A perceptually uniform color space,
# i.e. distances are meaningful. L* in [0..1] and a*, b* almost in [-1..1].
D65 = [0.95047, 1.00000, 1.08883]


def lab_f(t):
    return torch.where(t > 0.008856451679035631, t.pow(1/3), t * 7.787037037037035 + 0.13793103448275862)


def lab_finv(t):
    return torch.where(t > 0.20689655172413793, t.pow(3), 0.12841854934601665 * (t - 0.13793103448275862))


def lab2xyz(lab, wref=None):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if wref is None:
        wref = D65
    dim = 1 if len(lab.shape) == 4 else 0
    l, a, b = lab.chunk(3, dim=dim)
    
    l2 = (l + 0.16) / 1.16
    x = wref[0] * lab_finv(l2 + a/5)
    y = wref[1] * lab_finv(l2)
    z = wref[2] * lab_finv(l2 - b/2)
    xyz = torch.cat([x, y, z], dim=dim)

    return xyz


def xyz2lab(xyz, wref=None):
    """
    input xyz as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if wref is None:
        wref = D65
    dim = 1 if len(xyz.shape) == 4 else 0
    x, y, z = xyz.chunk(3, dim=dim)

    fy = lab_f(y / wref[1])
    l = 1.16 * fy - 0.16
    a = 5.0 * (lab_f(x/wref[0]) - fy)
    b = 2.0 * (fy - lab_f(z/wref[2]))
    xyz = torch.cat([l, a, b], dim=dim)

    return xyz


# composed functions
def lab2rgb(lab):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    return xyz2rgb(lab2xyz(lab))


def rgb2lab(rgb):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    return xyz2lab(rgb2xyz(rgb))


# tests
def test_implementation():
    lab = np.zeros((32, 32, 32, 3, 4))
    for r in range(0, 32):
        for g in range(0, 32):
            for b in range(0, 32):
                img = 8 * np.array([[[r, g, b]]]).astype(np.uint8)
                lab[r, g, b, :, 0] = color.rgb2lab(img)
                img_tensor = to_tensor(img)  # converts to float in range[0 .. 1]
                lab[r, g, b, :, 1] = rgb2lab(img_tensor).permute(1, 2, 0).numpy()
                lab[r, g, b, :, 3] = (lab2rgb(rgb2lab(img_tensor)) - img_tensor).permute(1, 2, 0).numpy()
    # residual
    # rescalling necessarry: skimage range ca. [0 .. 100] and [-100 .. 100],
    # own implementation ca. [0 .. 1] and [-1 .. 1]
    lab[:, :, :, :, 2] = lab[:, :, :, :, 1] - lab[:, :, :, :, 0] / 100
    print('# out of tollerance eps:')
    for e in range(-7, 0, 1):
        eps = 10**e
        indices_wrong_calculation = (np.abs(lab[:, :, :, :, 2]) > eps).nonzero()
        print('eps ', eps, '\t', len(indices_wrong_calculation[0]), ' / ', lab[:, :, :, :, 2].size)
    print('# value ranges:')
    print('skimage \t min =', np.min(lab[:, :, :, :, 0]), ' \t max =', np.max(lab[:, :, :, :, 0]))
    print('own \t\t min =', np.min(lab[:, :, :, :, 1]), ' \t max =', np.max(lab[:, :, :, :, 1]))
    
    print('residual back converted =', np.max(np.abs(lab[:, :, :, :, 3])))
    
    # hard condition: residual < 1e-4 relative to own implementation
    assert len((np.abs(lab[:, :, :, :, 2]) > 1e-4).nonzero()[0]) == 0
    assert np.max(np.abs(lab[:, :, :, :, 3])) < 1e-4
    
    return


def test_implementation_with_example():
    img = io.imread('https://poopr.org/images/2017/08/22/91615172-find-a-lump-on-cats-skin-632x475.jpg')
    
    lab_skimage = color.rgb2lab(img) / 100
    lab_own = rgb2lab(to_tensor(img)).permute(1, 2, 0).numpy()
   
    residual = lab_own - lab_skimage
    print('residual example image skimage vs own: min =', np.min(residual), ' \t max =', np.max(residual))
    
    plt.figure()
    plt.title('original image rgb')
    plt.imshow(img)
    plt.axis('off')

    plt.figure()
    plt.title('lab own')
    plt.imshow(lab_own)
    plt.axis('off')

    plt.figure()
    plt.title('lab skimage')
    plt.imshow(lab_skimage)
    plt.axis('off')
    
    plt.figure()
    plt.title('lab residual')
    plt.imshow(residual)
    plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    import numpy as np
    from skimage import color
    from torchvision.transforms.functional import to_tensor
    
    from skimage import io
    import matplotlib.pyplot as plt

    test_implementation()
    test_implementation_with_example()
