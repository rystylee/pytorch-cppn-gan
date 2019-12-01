import cv2

import torch

from src.resnet_cppn.models import Generator
from utils import get_coordinates, interpolate


class Tester(object):
    def __init__(self, config):
        self.config = config

        self.device = config.device
        self.max_itr = config.max_itr
        self.batch_size = config.batch_size
        self.img_size = config.img_size
        self.dim_z = config.dim_z
        self.dim_x = config.dim_x
        self.dim_y = config.dim_y
        self.dim_c = config.dim_c
        self.scale = config.scale

        self.generator = Generator(config).to(config.device)
        self.generator.eval()
        self._load_models(config.checkpoint_path)

        self.x, self.y, self.r = get_coordinates(self.dim_x, self.dim_y, self.scale, 1)
        self.x, self.y, self.r = self.x.to(self.device), self.y.to(self.device), self.r.to(self.device)

    def test(self):
        src_z = self._sample_z()
        dst_z = self._sample_z()

        counter = 0.0
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            with torch.no_grad():
                z = interpolate(src_z, dst_z, counter)

                out = self.generator(z, self.x, self.y, self.r)
                out = out.view(-1, self.dim_x, self.dim_y, self.dim_c).permute((0, 3, 1, 2))
                out = out.squeeze(0)
                out = (out + 1.0) * 0.5

                out = out.cpu().numpy().transpose(1, 2, 0)
                out = out[:, :, ::-1]
                cv2.imshow('Result', out)

                counter += 0.005
                if counter >= 1.0:
                    counter = 0.0
                    src_z = dst_z
                    dst_z = self._sample_z()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _sample_z(self):
        return torch.bmm(
            torch.ones(1, self.dim_x * self.dim_y, 1),
            torch.randn(1, 1, self.dim_z)
            ).to(self.device)

    def _load_models(self, model_state_path):
        checkpoint = torch.load(model_state_path)
        self.generator.load_state_dict(checkpoint['generator'])
        print('Loaded pretrained models...\n')
