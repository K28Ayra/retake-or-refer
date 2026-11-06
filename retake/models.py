import os

# weights go next to the data, not into the home folder
os.environ.setdefault("HF_HOME", "data/hf_cache")

import timm
import torch


def create_model(name, n_outputs, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=n_outputs)


def count_params(model):
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
    return total


def float_size_mb(model):
    """Size of the weights at 32 bits. The quantised size is measured later."""
    return count_params(model) * 4 / 1e6


def pick_gpu(min_free_gb=4.0):
    """The GPUs are shared, so take whichever has the most room right now. None if neither fits."""
    best_gpu = None
    best_free = 0.0
    for gpu in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(gpu)
        free_gb = free_bytes / 1024**3
        if free_gb > best_free:
            best_gpu = gpu
            best_free = free_gb
    if best_free < min_free_gb:
        return None, best_free
    return best_gpu, best_free
