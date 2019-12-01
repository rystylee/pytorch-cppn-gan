import os
from tqdm import tqdm

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from src.dcgan_cppn.models import Generator, Discriminator
from losses import GANLoss
from utils import get_coordinates, endless_dataloader

from data_loader import DataLoader


class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.device = config.device
        self.max_itr = config.max_itr
        self.batch_size = config.batch_size
        self.img_size = config.img_size
        self.dim_z = config.dim_z
        self.dim_c = config.dim_c
        self.scale = config.scale
        self.n_gen = config.n_gen

        self.start_itr = 1

        dataloader = DataLoader(
            config.data_root, config.dataset_name, config.img_size, config.batch_size, config.with_label
            )
        train_loader, test_loader = dataloader.get_loader(only_train=True)
        self.dataloader = train_loader
        self.dataloader = endless_dataloader(self.dataloader)

        self.generator = Generator(config).to(config.device)
        self.discriminator = Discriminator(config).to(config.device)

        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
        self.criterion = GANLoss()

        if not self.config.checkpoint_path == '':
            self._load_models(self.config.checkpoint_path)

        self.x, self.y, self.r = get_coordinates(self.img_size, self.img_size, self.scale, self.batch_size)
        self.x, self.y, self.r = self.x.to(self.device), self.y.to(self.device), self.r.to(self.device)

        self.writer = SummaryWriter(log_dir=config.log_dir)

    def train(self):
        print('Start training!\n')
        with tqdm(total=self.config.max_itr + 1 - self.start_itr) as pbar:
            for n_itr in range(self.start_itr, self.config.max_itr + 1):
                pbar.set_description(f'iteration [{n_itr}]')

                # ------------------------------------------------
                # Train G
                # ------------------------------------------------
                total_loss_g = 0
                for _ in range(self.n_gen):
                    self.optim_g.zero_grad()
                    z = torch.bmm(
                        torch.ones(self.batch_size, self.img_size * self.img_size, 1),
                        torch.randn(self.batch_size, 1, self.dim_z)
                        ).to(self.device)

                    fake_img = self.generator(z, self.x, self.y, self.r)
                    fake_img = fake_img.view(-1, self.img_size, self.img_size, self.dim_c).permute((0, 3, 1, 2))
                    d_fake = self.discriminator(fake_img)
                    loss_g = self.criterion(d_fake, 'g')
                    total_loss_g += loss_g.item()
                    loss_g.backward()
                    self.optim_g.step()
                total_loss_g /= float(self.n_gen)

                # ------------------------------------------------
                # Train D
                # ------------------------------------------------
                total_loss_d = 0
                total_loss_d_real = 0
                total_loss_d_fake = 0
                self.optim_d.zero_grad()
                img, label = next(self.dataloader)
                real_img, real_label = img.to(self.device), label.to(self.device)

                z = torch.bmm(
                    torch.ones(self.batch_size, self.img_size * self.img_size, 1),
                    torch.randn(self.batch_size, 1, self.dim_z)
                    ).to(self.device)
                with torch.no_grad():
                    fake_img = self.generator(z, self.x, self.y, self.r)
                    fake_img = fake_img.view(-1, self.img_size, self.img_size, self.dim_c).permute((0, 3, 1, 2))
                d_real = self.discriminator(real_img)
                d_fake = self.discriminator(fake_img.detach())
                loss_d_real = self.criterion(d_real, 'd_real')
                loss_d_fake = self.criterion(d_fake, 'd_fake')
                loss_d = loss_d_real + loss_d_fake
                total_loss_d += loss_d.item()
                total_loss_d_fake += loss_d_fake.item()
                total_loss_d_real += loss_d_real.item()
                loss_d.backward()
                self.optim_d.step()

                if n_itr % self.config.checkpoint_interval == 0:
                    self._save_models(n_itr)

                if n_itr % self.config.log_interval == 0:
                    tqdm.write('iteration: {}/{}, loss_g: {}, loss_d: {}, loss_d_real: {}, loss_d_fake: {}'.format(
                            n_itr, self.config.max_itr, total_loss_g, total_loss_d, total_loss_d_real, total_loss_d_fake))
                    self.writer.add_scalar('loss/loss_g', total_loss_g, n_itr)
                    self.writer.add_scalar('loss/loss_d', total_loss_d, n_itr)
                    self.writer.add_scalar('loss/loss_d_real', total_loss_d_real, n_itr)
                    self.writer.add_scalar('loss/loss_d_fake', total_loss_d_fake, n_itr)

                if n_itr % self.config.sample_interval == 0:
                    fake_name = f'fake_{n_itr}.jpg'
                    fake_path = os.path.join(self.config.sample_dir, fake_name)
                    torchvision.utils.save_image(fake_img.detach(), fake_path, normalize=True, range=(-1.0, 1.0))
                    real_name = f'real_{n_itr}.jpg'
                    real_path = os.path.join(self.config.sample_dir, real_name)
                    torchvision.utils.save_image(real_img, real_path, normalize=True, range=(-1.0, 1.0))

                pbar.update()

    def _load_models(self, model_state_path):
        checkpoint = torch.load(model_state_path)
        self.start_itr = checkpoint['n_itr'] + 1
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optim_g.load_state_dict(checkpoint['optim_g'])
        self.optim_d.load_state_dict(checkpoint['optim_d'])
        print(f'start_itr: {self.start_itr}')
        print('Loaded pretrained models...\n')

    def _save_models(self, n_itr):
        checkpoint_name = f'{self.config.dataset_name}-{self.config.img_size}_model_ckpt_{n_itr}.pth'
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        torch.save({
            'n_itr': n_itr,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optim_g': self.optim_g.state_dict(),
            'optim_d': self.optim_d.state_dict(),
        }, checkpoint_path)
        tqdm.write(f'Saved models: n_itr_{n_itr}')
