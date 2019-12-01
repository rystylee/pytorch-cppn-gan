import torch
import torchvision

from src.basic_cppn.config import get_config
from src.basic_cppn.models import CPPN
from utils import get_coordinates


def main():
    config = get_config()
    print(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.device = device
    print(f'device: {config.device}')

    model = CPPN(config).to(device)

    x, y, r = get_coordinates(config.dim_x, config.dim_y, config.scale)
    x, y, r = x.to(config.device), y.to(config.device), r.to(config.device)

    z = torch.randn(1, config.dim_z).to(device)
    scale = torch.ones((config.dim_x * config.dim_y, 1)).to(config.device)
    z_scaled = torch.matmul(scale, z)

    result = model(z_scaled, x, y, r)
    result = result.view(-1, config.dim_x, config.dim_y, config.dim_c).cpu()
    result = result.permute((0, 3, 1, 2))
    torchvision.utils.save_image(torchvision.utils.make_grid(result), 'sample.jpg')


if __name__ == "__main__":
    main()
