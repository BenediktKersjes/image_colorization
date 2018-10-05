import torch
import torch.nn as nn

import pytorch_differential_color as pdc
from image_generator import generate_images
from inceptionresnetv2 import inceptionresnetv2


class CheapConvNet(nn.Module):
    def __init__(self, out_channels=2):
        super(CheapConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        if out_channels == 2:
            # if predict a* and b*, use tanh activation
            self.features.add_module('26', nn.Tanh())

    def forward(self, x):
        x = self.features(x)
        return x


class ColorfulImageColorization(nn.Module):
    def __init__(self, out_channels=256, to_rgb=False):
        super(ColorfulImageColorization, self).__init__()
        self.out_channels = out_channels
        self.to_rgb = to_rgb
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv7
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            # output
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)
        )
        if self.out_channels == 2:
            # if predict a* and b*, use tanh activation
            self.features.add_module(str(len(self.features)), nn.Tanh())
        self.features.add_module(str(len(self.features)), nn.Upsample(scale_factor=4))

    def forward(self, image):
        output = self.features(image)
        if self.to_rgb:
            if self.out_channels == 2:
                # output has channels for a and b
                output = pdc.lab2rgb(torch.cat((image, output), 1))
            else:
                # sample a and b first
                output = generate_images(image, output, is_target=False)
        return output


class LTBCNetwork(nn.Module):
    def __init__(self, num_classes=10, rescale_input=True):
        super(LTBCNetwork, self).__init__()

        self.num_classes = num_classes
        self.rescale_input = rescale_input

        self.low_level_features_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.global_features_conv_net = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.global_features_linear1_net = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )

        self.global_features_linear2_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.classifier_net = nn.Sequential(
            nn.Linear(512, self.num_classes)
            # todo: more than one layer?!?
        )

        self.mid_level_net = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fusion_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.colorization_net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def fusion_net(self, x_local, x_global):

        x_global.unsqueeze_(2).unsqueeze_(3)
        x_global_expanded = x_global.expand_as(x_local)
        x = torch.cat([x_local, x_global_expanded], dim=1)

        fused = self.fusion_layers(x)

        return fused

    def forward(self, image):
        if self.rescale_input:
            image = nn.functional.interpolate(image, size=(224, 224))

        # # classification path
        # downsample
        input_classi = nn.functional.interpolate(image, size=(224, 224))
        # low level features
        low_level_classi = self.low_level_features_net(input_classi)
        # global features
        global_conv = self.global_features_conv_net(low_level_classi)
        x = global_conv.view(-1, 7*7*512)
        assert image.size()[0] == x.size()[0]  # assert batch dimension correct
        pre_classi = self.global_features_linear1_net(x)
        classification = self.classifier_net(pre_classi)
        fusion_global = self.global_features_linear2_net(pre_classi)

        # # local path
        # low level features
        if (image.size()[2] == 224) and (image.size()[3] == 224):
            low_level_local = low_level_classi
        else:
            low_level_local = self.low_level_features_net(image)

        mid_level_local = self.mid_level_net(low_level_local)
        # fuse global and local features
        fused = self.fusion_net(mid_level_local, fusion_global)
        # colorize
        chrominance_half = self.colorization_net(fused)
        # upsample
        chrominance = nn.functional.interpolate(chrominance_half, scale_factor=2)

        return classification, chrominance


class FusionLayer(nn.Module):
    def __init__(self, depth_before_fusion=1257, depth_after_fusion=256):
        super(FusionLayer, self).__init__()
        self.after_fusion = nn.Sequential(
            nn.Conv2d(depth_before_fusion, depth_after_fusion, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_local, x_global):
        x_global.unsqueeze_(2).unsqueeze_(3)
        x_global_expanded = x_global.expand(-1, -1, *(x_local.size()[-2:]))
        concatenated = torch.cat([x_local, x_global_expanded], dim=1)
        fused = self.after_fusion(concatenated)
        return fused


class DeepKoalarization(nn.Module):
    def __init__(self, out_channels=2, depth_before_fusion=1257, depth_after_fusion=256, use_224=False, to_rgb=False):
        super(DeepKoalarization, self,).__init__()
        self.feature_extractor = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
        self.encoder = self._build_encoder()
        self.fusion = FusionLayer(depth_before_fusion, depth_after_fusion)
        self.decoder = self._build_decoder(depth_after_fusion, out_channels)
        self.use_224 = use_224
        self.to_rgb = to_rgb

    def forward(self, image):
        img_224 = nn.functional.interpolate(image, size=(224, 224)) if self.use_224 else image
        img_299 = nn.functional.interpolate(image, size=(299, 299)).expand(-1, 3, -1, -1)
        encoded = self.encoder(img_224)
        global_features = self.feature_extractor(img_299)
        fused = self.fusion(encoded, global_features)
        output = self.decoder(fused)
        if self.to_rgb:
            if self.out_channels == 2:
                # output has channels for a and b
                output = pdc.lab2rgb(torch.cat((image, output), 1))
            else:
                # sample a and b first
                output = generate_images(image, output, is_target=False)
        return output

    def _build_encoder(self):
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        return model

    def _build_decoder(self, depth_after_fusion, out_channels):
        model = nn.Sequential(
            nn.Conv2d(depth_after_fusion, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        )
        if out_channels == 2:
            # if predict a* and b*, use tanh activation
            model.add_module(str(len(model)), nn.Tanh())
        model.add_module(str(len(model)), nn.Upsample(scale_factor=2))
        return model


class FusionLayerNorm(FusionLayer):
    def __init__(self, depth_before_fusion=1257, depth_after_fusion=256):
        super(FusionLayerNorm, self).__init__(depth_before_fusion, depth_after_fusion)
        self.norm_layer = nn.BatchNorm2d(depth_after_fusion)
    
    def forward(self, x_local, x_global):
        fused = super(FusionLayerNorm, self).forward(x_local, x_global)
        normed = self.norm_layer(fused)
        return normed


class DeepKoalarizationNorm(nn.Module):
    def __init__(self, out_channels=2, depth_before_fusion=1257, depth_after_fusion=256, use_224=False, to_rgb=False):
        super(DeepKoalarizationNorm, self).__init__()
        self.feature_extractor = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
        self.encoder = self._build_encoder()
        self.fusion = FusionLayerNorm(depth_before_fusion, depth_after_fusion)
        self.decoder = self._build_decoder(depth_after_fusion, out_channels)
        self.use_224 = use_224
        self.to_rgb = to_rgb

    def forward(self, image):
        img_224 = nn.functional.interpolate(image, size=(224, 224)) if self.use_224 else image
        img_299 = nn.functional.interpolate(image, size=(299, 299)).expand(-1, 3, -1, -1)
        encoded = self.encoder(img_224)
        global_features = self.feature_extractor(img_299)
        fused = self.fusion(encoded, global_features)
        output = self.decoder(fused)
        if self.to_rgb:
            if self.out_channels == 2:
                # output has channels for a and b
                output = pdc.lab2rgb(torch.cat((image, output), 1))
            else:
                # sample a and b first
                output = generate_images(image, output, is_target=False)
        return output

    def _build_encoder(self):
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        return model

    def _build_decoder(self, depth_after_fusion, out_channels):
        model = nn.Sequential(
            nn.Conv2d(depth_after_fusion, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        )
        if out_channels == 2:
            # if predict a* and b*, use tanh activation
            model.add_module(str(len(model)), nn.Tanh())
        model.add_module(str(len(model)), nn.Upsample(scale_factor=2))
        return model


if __name__ == "__main__":
    from skimage import color, io

    # net = LTBCNetwork(num_classes=10, rescale_input=True)
    net = ColorfulImageColorization(out_channels=256)
    print(net)

    # img = misc.imread('C:/Users/Hannes Perrot/Pictures/USA CA/08_20_Pacific Malte Julian/P1180185.JPG')
    img = io.imread('https://poopr.org/images/2017/08/22/91615172-find-a-lump-on-cats-skin-632x475.jpg')
    lab = color.rgb2lab(img)
    grey = lab[:, :, 0]
    input_tensor = torch.from_numpy(grey).unsqueeze_(0).unsqueeze_(0).float()
    input_tensor = nn.functional.interpolate(input_tensor, size=(96, 96))
    print(net)
    print(input_tensor.size(), 'input_tensor')

    print(net(input_tensor).shape, 'output')
