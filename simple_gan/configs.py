import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


lr = 3e-4
z_dim = 64
img_dim = 28*28*1
batch_size = 32
epochs = 200



if __name__ == "__main__":
    print(get_device())