import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboard_logger import Logger

import pytorch_differential_color as pdc
from multinomial_cross_entropy_loss import MultinomialCrossEntropyLoss
from config import images_path, logs_path, grid_size, trained_models_path, dataset
from image_generator import generate_images_numpy
from network import DeepKoalarization, CheapConvNet, ColorfulImageColorization, DeepKoalarizationNorm
from perceptual_loss import PerceptualLoss
from colorization_dataset import ColorizationDataset, conversion_batch


class TrainNetwork:
    def __init__(self, batch_size=64, image_size=64,
                 load_model=None, iterations_start=0, seed=None,
                 epochs=10000, lr_decay_iter=250000,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3, weighting_factor=0.5,
                 regression=False, loss='MultiLabelSoftMarginLoss', network='ColorfulImageColorization',
                 convert_on_gpu=False, do_not_log=False):
        # hyperparameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.iterations = iterations_start
        self.seed = np.random.randint(0, 10000) if seed is None else seed
        self.epochs = epochs
        self.lr_decay_iter = lr_decay_iter
        self.lr0 = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.weighting_factor = weighting_factor
        self.lr = self.calc_learning_rate()
        self.regression = regression
        self.loss = loss
        self.model_name = '{date:%Y_%m_%d__%H_%M_%S}_{net}_{loss}_{dataset}'.format(
            date=datetime.datetime.now(), net=network, loss=self.loss, dataset=dataset)
        torch.manual_seed(self.seed)
        self.convert_on_gpu = convert_on_gpu

        # tensorboard
        if not do_not_log:
            self.logger = Logger(logs_path + self.model_name)
            self.log_hyperparameter()

        # model, loss, optimizer
        assert network in ['DeepKoalarization', 'CheapConvNet', 'ColorfulImageColorization', 'DeepKoalarizationNorm']
        
        if self.regression:
            out_channels = 2
        else:
            out_channels = int((256/grid_size)**2)

        if network == 'DeepKoalarization':
            self.model = DeepKoalarization(out_channels=out_channels, to_rgb=(self.loss == 'PerceptualLoss'))
        elif network == 'DeepKoalarizationNorm':
            self.model = DeepKoalarizationNorm(out_channels=out_channels, to_rgb=(self.loss == 'PerceptualLoss'))
        elif network == 'CheapConvNet':
            assert self.loss != 'PerceptualLoss'
            self.model = CheapConvNet(out_channels=out_channels)
        elif network == 'ColorfulImageColorization':
            self.model = ColorfulImageColorization(out_channels=out_channels, to_rgb=(self.loss == 'PerceptualLoss'))

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps,
                                    weight_decay=self.weight_decay)

        assert loss in ['PerceptualLoss', 'MSELoss', 'MultiLabelSoftMarginLoss', 'BCEWithLogitsLoss',
                        'MultinomialCrossEntropyLoss']

        if loss == 'PerceptualLoss':
            self.loss_fn = PerceptualLoss()
        elif loss == 'MSELoss':
            assert self.regression
            self.loss_fn = nn.MSELoss()
        elif loss == 'MultiLabelSoftMarginLoss':
            assert not self.regression
            w = torch.load(images_path+'classification_weights_{}_{}.pth'.format(grid_size, self.weighting_factor))
            self.loss_fn = nn.MultiLabelSoftMarginLoss(weight=w.view(-1, 1, 1))
        elif loss == 'BCEWithLogitsLoss':
            assert not self.regression
            w = torch.load(images_path+'classification_weights_{}_{}.pth'.format(grid_size, self.weighting_factor))
            self.loss_fn = nn.BCEWithLogitsLoss(weight=w.view(-1, 1, 1))
        elif loss == 'MultinomialCrossEntropyLoss':
            assert not self.regression
            w = torch.load(images_path + 'classification_weights_{}_{}.pth'.format(grid_size, self.weighting_factor))
            self.loss_fn = MultinomialCrossEntropyLoss(weights=w)

        # load model
        if load_model is not None:
            # Load pre learned AlexNet with changed number of output classes
            state_dict = torch.load(trained_models_path+load_model, map_location='cpu')
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.adjust_learning_rate()
        
        # Use cuda if available
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()
            self.loss_fn.cuda()

            if load_model is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        # Load dataset
        kwargs = {'num_workers': 8, 'pin_memory': True} if self.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            ColorizationDataset(
                images_path, 
                train=True, 
                size=(self.image_size, self.image_size), 
                target_rgb=(loss == 'PerceptualLoss'),
                convert_to_categorical=(not self.regression),
                do_not_convert=convert_on_gpu),
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.test_loader = torch.utils.data.DataLoader(
            ColorizationDataset(
                images_path, 
                train=False, 
                size=(self.image_size, self.image_size), 
                target_rgb=(loss == 'PerceptualLoss'),
                convert_to_categorical=(not self.regression),
                do_not_convert=convert_on_gpu), 
            batch_size=8,
            drop_last=True,
            shuffle=True,
            **kwargs)
        self.test_iterator = iter(self.test_loader)

    def calc_learning_rate(self):
        """
        Reduce the learning rate by factor 0.5 every lr_decay_iter
        :return: None
        """
        lr = self.lr0 * (0.1 ** (self.iterations // self.lr_decay_iter))
        return lr

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def reduce_learning_rate(self):
        lr = self.calc_learning_rate()

        if abs(lr - self.lr) > 1e-7:
            self.lr = lr
            self.adjust_learning_rate()

    def train(self):
        """
        Train the model for one epoch and save the result as a .pth file
        :return: None
        """
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            train_loss_epoche = 0
            batch_start_time = time.clock()
            batch_idx = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.reduce_learning_rate()
                train_loss_epoche += self.train_one_iter(data, target, epoch, batch_idx)
                self.iterations += 1
                print('Batch ' + str(batch_idx + 1) + ' took ' + str(time.clock() - batch_start_time) + ' seconds')
                batch_start_time = time.clock()

            # Print information about current epoch
            train_loss_epoche /= (batch_idx + 1)
            print('Train Epoch: {} \tAverage loss: {:.6f}'
                  .format(epoch, train_loss_epoche))

    def train_one_iter(self, data, target, epoch, batch_idx):

        if self.cuda:
            data = data.cuda()
            target = target.cuda()

        if self.convert_on_gpu:
            lab = pdc.rgb2lab(data.float()/255)
            data, target = torch.split(lab, [1, 2], dim=1)
            if not self.regression:
                target = conversion_batch(target)

        # Optimize using backpropagation
        self.optimizer.zero_grad()
        output = self.model(data)

        loss = self.loss_fn(output, target)
        loss.backward()
        self.optimizer.step()

        # Print information about current step
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
            epoch, batch_idx + 1, len(self.train_loader), loss.item()))
        if self.iterations % 5 == 0:
            # log loss
            test_data, test_target = self.get_next_test_batch()
            self.model.eval()
            test_output = self.model(test_data)
            self.model.train()
            test_loss = self.loss_fn(test_output, test_target)
            self.log_scalars(self.iterations, loss, test_loss)

            if self.iterations % 50 == 0:
                # log images
                self.log_images(self.iterations, test_data, test_target, test_output, data, target, output)
        if self.iterations % 1000 == 0 and self.iterations > 0:
            # Save snapshot
            model_name = self.model_name + '_iter{}'.format(self.iterations)
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, trained_models_path+'{}.pth'.format(model_name))
        if self.iterations % 600 == 0:
            # log stuff
            self.log_values_gradients(self.iterations)

        return loss.item()

    def log_hyperparameter(self):
        info = {'batch_size': self.batch_size,
                'image_size': self.image_size,
                'seed': self.seed,
                'epochs': self.epochs,
                'learning_decay_iter': self.lr_decay_iter,
                'learning_rate_0': self.lr0,
                'betas[0]': self.betas[0],
                'betas[1]': self.betas[1],
                'eps_optimizer': self.eps,
                'weight_decay_optimizer': self.weight_decay,
                'weighting_factor': self.weighting_factor,
                'regression': self.regression,
                'convert_on_gpu': self.convert_on_gpu}

        for tag, value in info.items():
            self.logger.log_value(tag, value, 0)

    def log_scalars(self, step, train_loss, test_loss):
        # adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py
        # 1. Log scalar values (scalar summary)
        info = {'train_loss': train_loss.item(),
                'test_loss': test_loss.item(),
                'learning_rate': self.lr}

        for tag, value in info.items():
            self.logger.log_value(tag, value, step)

    def log_images(self, step, test_data, test_target, test_output, train_data, train_target, train_output):
        # 3. Log test images (image summary)
        num_images = 1
        test_original = self.convert_to_images(test_data[:num_images], test_target[:num_images], is_target=True)
        test_colorized = self.convert_to_images(test_data[:num_images], test_output[:num_images], is_target=False)
        train_original = self.convert_to_images(train_data[:num_images], train_target[:num_images], is_target=True)
        train_colorized = self.convert_to_images(train_data[:num_images], train_output[:num_images], is_target=False)
        info = {'test colorized': test_colorized,
                'test original': test_original,
                'train colorized': train_colorized,
                'train original': train_original}

        for tag, images in info.items():
            self.logger.log_images(tag, images, step)

    def log_values_gradients(self, step):
        # adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py
        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.model.named_parameters():
            if 'feature_extractor' not in tag:
                tag = tag.replace('.', '/')
                self.logger.log_histogram(tag, value.data.cpu().numpy(), step)
                self.logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), step)

    def convert_to_images(self, data, output_or_target, is_target, t=0.38):
        if self.loss == 'PerceptualLoss':
            return output_or_target.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float64)
        else:
            return generate_images_numpy(data, output_or_target, is_target, regression=self.regression, t=t)

    def get_next_test_batch(self):
        try:
            data, target = next(self.test_iterator)
        except StopIteration:
            self.test_iterator = iter(self.test_loader)
            data, target = next(self.test_iterator)

        if self.cuda:
            data = data.cuda()
            target = target.cuda()

        if self.convert_on_gpu:
            lab = pdc.rgb2lab(data.float()/255)
            data, target = torch.split(lab, [1, 2], dim=1)
            if not self.regression:
                target = conversion_batch(target)

        return data, target


if __name__ == '__main__':
    init_start_time = time.clock()
    trainer = TrainNetwork(
        batch_size=13,
        image_size=128,
        load_model=None,
        iterations_start=0,
        lr=1e-4,
        lr_decay_iter=1000000,
        regression=False,
        loss='MultinomialCrossEntropyLoss',
        network='DeepKoalarizationNorm',
        convert_on_gpu=True,
        weighting_factor=0.5)
    print('Initialization took ' + str(time.clock() - init_start_time) + ' seconds')
    trainer.train()
