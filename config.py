import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = "/data1/linkdom/data"
seed = 1337
batch_size = 512
learning_rate = 3e-4
epochs = 5
