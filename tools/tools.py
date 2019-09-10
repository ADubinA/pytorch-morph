import torch

def batch_duplication(tensor, batch_size):
    # tensor = tensor.unsqueeze(0)
    return torch.cat(list(tensor for _ in range(batch_size)))
