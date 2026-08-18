import os
import torch
import torch.nn as nn
from Discriminator import Discriminator
from Generator import Generator

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def save_checkpoint(
    name, epoch, step, gen, disc, opt_gen, opt_disc,
    configs=None, checkpoint_dir="checkpoints",
):
    """Save both networks and both optimizers -- resuming a GAN needs all four,
    since Adam's momentum buffers are part of the training state.

    `configs` may be the configs module; its uppercase names are snapshotted so
    the checkpoint stays interpretable after configs.py moves on.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "gen": gen.state_dict(),
            "disc": disc.state_dict(),
            "opt_gen": opt_gen.state_dict(),
            "opt_disc": opt_disc.state_dict(),
            "configs": {} if configs is None else {
                k: v for k, v in vars(configs).items() if k.isupper()
            },
        },
        path,
    )
    print(f"  -> saved {path}")
    return path


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert(disc(x).shape == (N, 1, 1, 1))

    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert(gen(z).shape == (N, in_channels, H, W))

    print("success")


if __name__=="__main__":
    test()