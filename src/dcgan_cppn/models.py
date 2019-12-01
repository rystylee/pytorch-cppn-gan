import torch.nn as nn


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        dim_z = config.dim_z
        dim_c = config.dim_c
        ngf = config.ngf

        self.l_z = nn.Linear(dim_z, ngf)
        self.l_x = nn.Linear(1, ngf, bias=False)
        self.l_y = nn.Linear(1, ngf, bias=False)
        self.l_r = nn.Linear(1, ngf, bias=False)

        self.ln_seq = nn.Sequential(
            nn.Tanh(),

            nn.Linear(ngf, ngf),
            nn.Tanh(),

            nn.Linear(ngf, ngf),
            nn.Tanh(),

            nn.Linear(ngf, ngf),
            nn.Tanh(),

            nn.Linear(ngf, dim_c),
            nn.Tanh()
            )

        self._initialize()

    def _initialize(self):
        self.apply(weights_init_g)

    def forward(self, z, x, y, r):
        u = self.l_z(z) + self.l_x(x) + self.l_y(y) + self.l_r(r)
        out = self.ln_seq(u)
        return out


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        dim_c = config.dim_c
        ndf = config.ndf

        self.net = nn.Sequential(
            nn.Conv2d(dim_c, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            )

        self._initialize()

    def _initialize(self):
        self.apply(weights_init_d)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1)
        return x
