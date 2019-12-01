import torch

from src.resnet_cppn.config import get_config
from src.resnet_cppn.trainer import Trainer
from src.resnet_cppn.tester import Tester


def main():
    config = get_config()
    print(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.device = device
    print(f'device: {config.device}')

    if config.mode == 'train':
        torch.backends.cudnn.benchmark = True
        trainer = Trainer(config)
        trainer.train()
    elif config.mode == 'test':
        tester = Tester(config)
        tester.test()


if __name__ == "__main__":
    main()
