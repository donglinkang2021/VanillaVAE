import torch
from utils import (
    get_datasets, 
    get_loader, 
    set_seed, 
    evaluate, 
    plot_latent_codes, 
    plot_reconstructed_manifold
)
from model import AutoEncoder, VariationalAutoEncoder
import config

set_seed(config.seed)

def _test(model: AutoEncoder) -> None:
    batch_size = 512
    train_dataset, test_dataset = get_datasets(config.data_root)
    _, test_loader = get_loader(train_dataset, test_dataset, batch_size)
    metrics = evaluate(model, test_loader)
    print(metrics)
    plot_latent_codes(model, test_loader, num_samples_to_plot=2000)
    plot_reconstructed_manifold(model, num_images_per_dim=12)

def _test_ae():
    device = config.device
    model_path = "ckpts/ae_5epo.pth"
    model = AutoEncoder(latent_dims = 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    _test(model)

def _test_vae():
    device = config.device
    model_path = "ckpts/vae_5epo.pth"
    model = VariationalAutoEncoder(latent_dims = 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    _test(model)

def main():
    _test_ae()
    _test_vae()

if __name__ == "__main__":
    main()
