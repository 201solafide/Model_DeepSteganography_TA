import torch
import os

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Menyimpan state_dict model dan optimizer.
    Args:
        state (dict): Dictionary yang berisi state model, optimizer, dan epoch.
        filename (str): Nama file untuk menyimpan checkpoint.
    """
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    """
    Memuat checkpoint dari file.
    Args:
        filename (str): Nama file checkpoint.
        model (nn.Module): Model PyTorch yang akan dimuat statenya.
        optimizer (torch.optim.Optimizer): Optimizer yang akan dimuat statenya.
    Returns:
        tuple: (model, optimizer, start_epoch)
    """
    if os.path.isfile(filename):
        print(f"=> Memuat checkpoint dari '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"=> Berhasil memuat checkpoint (epoch {start_epoch})")
        return model, optimizer, start_epoch
    else:
        print(f"=> Tidak ada checkpoint ditemukan di '{filename}'")
        return model, optimizer, 0