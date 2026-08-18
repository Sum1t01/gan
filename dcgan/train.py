import torch
import torchvision
import torch.nn as nn
from Discriminator import Discriminator
from Generator import Generator
from utils import initialize_weights, save_checkpoint
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from  torch.utils.data import DataLoader
import torch.optim as optim
import configs
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = configs.get_device()


transforms = transforms.Compose(
    [
        transforms.Resize(configs.IMAGE_SIZE),
        transforms.CenterCrop(configs.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(configs.CHANNELS_IMG)], [0.5 for _ in range(configs.CHANNELS_IMG)],
        )
    ]
)

# datasets = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
datasets = datasets.ImageFolder(root="celeb_dataset", transform=transforms)

loader = DataLoader(dataset=datasets, batch_size=configs.BATCH_SIZE, shuffle=True)

gen = Generator(configs.Z_DIM, configs.CHANNELS_IMG, configs.FEATURES_GEN).to(device)
disc = Discriminator(configs.CHANNELS_IMG, configs.FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=configs.LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=configs.LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, configs.Z_DIM, 1, 1).to(device)

writer_fake = SummaryWriter("logs/fake", flush_secs=5)
writer_real = SummaryWriter("logs/real", flush_secs=5)

step = 0

gen.train()
disc.train()

def train():
    global step
    for epoch in range(configs.NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            noise = torch.randn(real.shape[0], configs.Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            #Train Discriminator: max log(D(real)) + log(1- D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            #Train Generator: max log(D(G(z)))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if(batch_idx % 100 == 0):
                print(
                    f"Epoch: [{epoch}/{configs.NUM_EPOCHS}] BATCH {batch_idx}/{len(loader)} "
                    f"Loss D: {loss_disc: .4f} Loss G: {loss_gen: .4f}"
                )
            
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    image_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    image_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            
                    writer_fake.add_image(
                        "Fake: ", image_grid_fake, global_step=step
                    )
            
                    writer_real.add_image(
                        "Real: ", image_grid_real, global_step=step
                    )
            
                    step+=1



        

        # Periodic checkpoint: after every 10th completed epoch.
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                f"epoch_{epoch + 1:03d}", epoch + 1, step,
                gen, disc, opt_gen, opt_disc, configs,
            )

    # Final model, timestamped so consecutive runs never overwrite each other.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_checkpoint(
        f"final_{stamp}", configs.NUM_EPOCHS, step,
        gen, disc, opt_gen, opt_disc, configs,
    )


if __name__ == "__main__":
    train()


        

