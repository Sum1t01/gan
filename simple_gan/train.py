import torch
import torchvision.transforms as transforms
from Discriminator import Discriminator
from Generator import Generator
import configs
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

device = configs.get_device()

disc = Discriminator(configs.img_dim).to(device)
gen = Generator(configs.z_dim, configs.img_dim).to(device)

fixed_noise = torch.randn(configs.batch_size, configs.z_dim).to(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=configs.batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=configs.lr)
opt_gen = optim.Adam(gen.parameters(), lr=configs.lr)
criterion = nn.BCELoss()


writer_fake = SummaryWriter("runs/GAN_MNIST/fake", flush_secs=5)
writer_real = SummaryWriter("runs/GAN_MNIST/real", flush_secs=5)

step = 0

def train():
    global step
    for epoch in range(configs.epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            #Train Discriminator: max log(D(real)) + log(1- D(G(z)))
            noise = torch.randn(batch_size, configs.z_dim).to(device)
            fake = gen(noise)

            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            #Train Generator: max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if(batch_idx == 0):
                print(
                    f"Epoch: [{epoch}/{configs.epochs}] "
                    f"Loss D: {lossD: .4f} Loss G: {lossG: .4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    image_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    image_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Mnist fake Images: ", image_grid_fake, global_step=step
                    )

                    writer_real.add_image(
                        "Mnist real Images: ", image_grid_real, global_step=step
                    )

                    step+=1




if __name__ == "__main__":
    train()