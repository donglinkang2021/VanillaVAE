import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import AutoEncoder, VariationalAutoEncoder
import config

def set_seed(seed):
    """make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_datasets(root:str):
    transform = transforms.ToTensor()
    train_dataset = MNIST(root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_loader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(
        model:AutoEncoder, 
        train_loader:DataLoader, 
        optimizer:torch.optim.Optimizer, 
    ) -> None:
    device = config.device
    model.train()
    # Define a more detailed bar format
    pbar = tqdm(train_loader, desc='Training', dynamic_ncols=True, leave=False)
    for data, _ in pbar: # Iterate directly over pbar
        data = data.to(device)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

@torch.no_grad()
def evaluate(
        model:AutoEncoder, 
        test_loader:DataLoader, 
    ) -> dict:
    device = config.device
    model.eval()
    metrics = {}
    running_loss = 0.0
    # Define a more detailed bar format
    pbar = tqdm(test_loader, desc='Evaluating', dynamic_ncols=True, leave=False)
    for data, _ in pbar: # Iterate directly over pbar
        data = data.to(device)
        loss = model(data)
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()}) # Postfix for individual batch loss during evaluation
    metrics['test loss'] = running_loss / len(test_loader)
    return metrics

# -----------------------------------plot------------------------------------
@torch.no_grad()
def plot_latent_codes(
    model: AutoEncoder, 
    data_loader: DataLoader, 
    num_samples_to_plot: int = 1000
) -> None:
    device = config.device
    model.eval()
    latents_list = []
    labels_list = []
    
    # tqdm setup for manual updates
    effective_total = min(num_samples_to_plot, len(data_loader.dataset))
    pbar = tqdm(total=effective_total, desc='Plotting Latents', dynamic_ncols=True, leave=False)
    
    processed_samples = 0
    for data, labels in data_loader:
        if processed_samples >= num_samples_to_plot:
            break
        
        samples_in_batch = data.size(0)
        samples_to_process = min(samples_in_batch, num_samples_to_plot - processed_samples)
        
        data = data[:samples_to_process].to(device)
        labels = labels[:samples_to_process] # Ensure labels match the processed data

        latent_codes = model.encode(data)
        if isinstance(latent_codes, tuple):
            latent_codes = latent_codes[0]
        latents_list.append(latent_codes.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        
        processed_samples += samples_to_process
        pbar.update(samples_to_process)
        
    pbar.close() # Close manually managed pbar

    if not latents_list: # Handle case where no samples were processed
        print("No samples processed for plotting.")
        return

    latents_array = np.concatenate(latents_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    # This check is now less likely to be needed due to controlled sampling, but good for safety
    if latents_array.shape[0] > num_samples_to_plot:
        indices = np.random.choice(latents_array.shape[0], num_samples_to_plot, replace=False)
        latents_array = latents_array[indices]
        labels_array = labels_array[indices]

    if latents_array.shape[1] != 2:
        print(f"Latent space dimension is {latents_array.shape[1]}. Plotting first 2 dimensions.")
        if latents_array.shape[1] < 2:
            print("Cannot plot, latent dimension is less than 2.")
            return

    plt.figure(figsize=(4, 3))
    scatter = plt.scatter(latents_array[:, 0], latents_array[:, 1], c=labels_array, cmap='tab10', s=10)
    plt.colorbar(scatter, label='Digit Label')
    if isinstance(model, VariationalAutoEncoder):
        model_name = "VAE"
    else:
        model_name = "AE"
    # plt.title(f'{model_name} Latent Space Visualization of MNIST Digits')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    save_path = f"plots/{model_name.lower()}_latent_space.png"
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close the plot to free memory
    print(f"Latent space plot saved to {save_path}")
    # plt.show() # Uncomment if you want to display the plot interactively


@torch.no_grad()
def plot_reconstructed_manifold(
    model: AutoEncoder,
    num_images_per_dim: int = 10, # Number of images per dimension in the grid
) -> None:
    """
    Generates and plots a manifold of images by sampling a 2D latent space.
    """
    device = config.device
    model.eval()

    if isinstance(model, VariationalAutoEncoder):
        r0 = (-2, 2)  # Range for latent dimension 1
        r1 = (-2, 2)  # Range for latent dimension 2
    else:
        r0 = (-5, 10)  # Range for latent dimension 1
        r1 = (0, 15)  # Range for latent dimension 2

    w = 28  # Image width/height for MNIST
    img_grid_shape = (num_images_per_dim * w, num_images_per_dim * w)
    img_grid = np.zeros(img_grid_shape)
    
    # Create coordinates for sampling the latent space
    x_coords = np.linspace(r0[0], r0[1], num_images_per_dim)
    y_coords = np.linspace(r1[0], r1[1], num_images_per_dim) # y_coords from r1_min to r1_max
    
    pbar = tqdm(total=num_images_per_dim * num_images_per_dim, desc='Generating manifold', leave=False)

    for i, y_val in enumerate(y_coords): # i iterates from 0 to N-1
        for j, x_val in enumerate(x_coords): # j iterates from 0 to N-1
            # Create a 2D latent vector sample
            z_sample = torch.tensor([[x_val, y_val]], dtype=torch.float32).to(device)
            
            # Decode the latent vector to get an image
            # model.decode is expected to return a tensor of shape (1, 1, 28, 28)
            reconstructed_sample = model.decode(z_sample) 
            
            # Denormalize the image (assuming normalization was (0.5, 0.5))
            # reconstructed_sample_denorm = reconstructed_sample * 0.5 + 0.5
            image_np = reconstructed_sample.cpu().squeeze().numpy() # Shape (28, 28)
            
            # Fill into the image grid
            # The y-axis in imshow typically goes from top to bottom.
            # If y_coords[i] is ascending (y_min to y_max),
            # and we want y_min at the visual bottom of the plot,
            # we map y_coords[i] to grid_row = (num_images_per_dim - 1 - i).
            grid_row_start = (num_images_per_dim - 1 - i) * w
            grid_col_start = j * w
            img_grid[grid_row_start : grid_row_start + w, 
                     grid_col_start : grid_col_start + w] = image_np
            pbar.update(1)
    pbar.close()

    if isinstance(model, VariationalAutoEncoder):
        model_name = "VAE"
    else:
        model_name = "AE"
    plt.figure(figsize=(4, 3)) # Adjust figure size as needed
    # extent defines [left, right, bottom, top] in data coordinates.
    # origin='upper' (default) means img_grid[0,0] is at the top-left.
    # Our filling maps y_max to img_grid[0,:] and y_min to img_grid[N-1,:].
    plt.imshow(img_grid, extent=[*r0, *r1], cmap='viridis', origin='upper')
    plt.xlabel(f"Latent Dimension 1")
    plt.ylabel(f"Latent Dimension 2")
    # plt.title(f"{model_name} Reconstructed Image Manifold from 2D Latent Space")
    plt.colorbar(label="Pixel Intensity (denormalized)")
    plt.tight_layout()
    save_path = f"plots/{model_name.lower()}_reconstructed_manifold.png"
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close the plot to free memory
    print(f"Reconstructed manifold plot saved to {save_path}")

from PIL import Image
def interpolate_gif(
        model: AutoEncoder,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        num_steps: int = 10,
        filename: str = "interpolation.gif"
    ) -> torch.Tensor:
    z_1 = model.encode(x_1)
    z_2 = model.encode(x_2)
    if isinstance(z_1, tuple):
        z_1 = z_1[0]
        z_2 = z_2[0]

    z = torch.stack(
        [z_1 + (z_2 - z_1) * i / num_steps for i in range(num_steps + 1)]
    )

    interpolated_images = model.decode(z)
    interpolated_images = interpolated_images.cpu().detach().numpy() * 255
    interpolated_images = interpolated_images.astype(np.uint8)

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((64, 64)) for img in interpolated_images]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        filename,
        save_all=True,
        append_images=images_list[1:],
        loop=1
    )
