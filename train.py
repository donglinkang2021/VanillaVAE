import torch
from tqdm import tqdm
from model import AutoEncoder, VariationalAutoEncoder
from utils import set_seed, get_datasets, get_loader, train, evaluate
import config

set_seed(config.seed)

def _train(model:AutoEncoder) -> None:
    # training config
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    device = config.device
    model = model.to(device)
    train_dataset, test_dataset = get_datasets(config.data_root)
    train_loader, test_loader = get_loader(train_dataset, test_dataset, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training {model.__class__.__name__} for {epochs} epochs on {device}...")
    # Iterate directly over tqdm object for epochs
    epoch_pbar = tqdm(range(epochs), desc='Overall Progress', dynamic_ncols=True, leave=True)
    
    for epoch in epoch_pbar:
        train(model, train_loader, optimizer)
        metrics = evaluate(model, test_loader)
        # Update epoch progress bar postfix with evaluation metrics
        epoch_pbar.set_postfix(metrics) 
        # Explicitly print the test loss for the current epoch
        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Test Loss: {metrics['test loss']:.4f}")

def _train_ae():
    model = AutoEncoder(latent_dims = 2)
    # model.apply(init_weights) # future work
    _train(model)
    model_path = "ckpts/ae_5epo.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def _train_vae():
    model = VariationalAutoEncoder(latent_dims = 2)
    # model.apply(init_weights) # future work
    _train(model)
    model_path = "ckpts/vae_5epo.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    # _train_ae()
    _train_vae()

if __name__ == '__main__':
    main()

