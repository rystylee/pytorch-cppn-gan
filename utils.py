import torch
import numpy as np


def get_coordinates(dim_x, dim_y, scale=1.0, batch_size=1):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = dim_x * dim_y
    x_range = scale * (np.arange(dim_x) - (dim_x - 1) / 2.0) / (dim_x - 1) / 0.5
    y_range = scale * (np.arange(dim_y) - (dim_y - 1) / 2.0) / (dim_y - 1) / 0.5
    x_mat = np.matmul(np.ones((dim_y, 1)), x_range.reshape((1, dim_x)))
    y_mat = np.matmul(y_range.reshape((dim_y, 1)), np.ones((1, dim_x)))
    r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()


def endless_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def interpolate(src, dst, coef):
    return src + (dst - src) * coef
