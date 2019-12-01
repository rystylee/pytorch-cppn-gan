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


class ResBlock(nn.Module):
    def __init__(self, ch, n_block=4):
        super(ResBlock, self).__init__()

        self.modules1 = []
        for i in range(n_block):
            self.modules1.append(nn.Linear(ch, ch))
            self.modules1.append(nn.ReLU())
        self.net1 = nn.Sequential(*self.modules1)

        self.modules2 = []
        self.modules2.append(nn.Linear(ch, ch))
        self.modules2.append(nn.Tanh())
        self.net2 = nn.Sequential(*self.modules2)

        self._initialize()

    def _initialize(self):
        self.net1.apply(self._weights_init_module1)
        self.net2.apply(self._weights_init_module2)

    def _weights_init_module1(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def _weights_init_module2(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.001)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

    def residual(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

    def shortcut(self, x):
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        dim_c = config.dim_c
        ngf = config.ngf
        dim_z = config.dim_z

        self.l_z = nn.Linear(dim_z, ngf)
        self.l_x = nn.Linear(1, ngf, bias=False)
        self.l_y = nn.Linear(1, ngf, bias=False)
        self.l_r = nn.Linear(1, ngf, bias=False)

        res_blocks = []
        for i in range(config.n_resblock):
            res_blocks.append(ResBlock(ngf, config.n_sub_resblock))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.l_out = nn.Linear(ngf, dim_c)
        self.tanh = nn.Tanh()

        self._initialize()

    def _initialize(self):
        self.l_z.apply(weights_init_g)
        self.l_x.apply(weights_init_g)
        self.l_y.apply(weights_init_g)
        self.l_r.apply(weights_init_g)
        self.l_out.apply(weights_init_g)

    def forward(self, z, x, y, r):
        u = self.l_z(z) + self.l_x(x) + self.l_y(y) + self.l_r(r)
        h = self.tanh(u)
        h = self.res_blocks(u)
        h = self.l_out(h)
        h = self.tanh(h)
        return h


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
