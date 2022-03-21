import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_num_params(model):
    return sum([p.numel() for p in model.parameters()])
