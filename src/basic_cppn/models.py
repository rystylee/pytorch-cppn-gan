import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CPPN(nn.Module):
    def __init__(self, config):
        super(CPPN, self).__init__()
        dim_z = config.dim_z
        dim_c = config.dim_c
        ch = config.ch

        self.l_z = nn.Linear(dim_z, ch)
        self.l_x = nn.Linear(1, ch, bias=False)
        self.l_y = nn.Linear(1, ch, bias=False)
        self.l_r = nn.Linear(1, ch, bias=False)

        self.ln_seq = nn.Sequential(
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, dim_c),
            nn.Sigmoid()
            )

        self._initialize()

    def _initialize(self):
        self.apply(weights_init)

    def forward(self, z, x, y, r):
        u = self.l_z(z) + self.l_x(x) + self.l_y(y) + self.l_r(r)
        out = self.ln_seq(u)
        return out
