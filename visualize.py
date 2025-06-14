import torch
from model import AutoEncoder, VariationalAutoEncoder
from utils import (
    get_datasets, 
    get_loader, 
    set_seed, 
    interpolate_gif
)
import config

set_seed(config.seed)

def get_two_images(num_start:int, num_end:int):
    batch_size = 512
    train_dataset, test_dataset = get_datasets(config.data_root)
    _, test_loader = get_loader(train_dataset, test_dataset, batch_size)
    x, y = next(iter(test_loader))
    x_1 = x[y == num_start][1].to(config.device) # find a 1
    x_2 = x[y == num_end][1].to(config.device)
    return x_1, x_2

def _interpolate_ae():
    model_path = "ckpts/ae_5epo.pth"
    device = config.device
    model = AutoEncoder(latent_dims=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    x_1, x_2 = get_two_images(0, 9)
    interpolate_gif(model, x_1, x_2, num_steps=100, filename="plots/ae.gif")

def _interpolate_vae():
    model_path = "ckpts/vae_5epo.pth"
    device = config.device
    model = VariationalAutoEncoder(latent_dims=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    x_1, x_2 = get_two_images(0, 9)
    interpolate_gif(model, x_1, x_2, num_steps=100, filename="plots/vae.gif")

def main():
    # _interpolate_ae()
    _interpolate_vae()

if __name__ == '__main__':
    main()
