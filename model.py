import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, image_dim, conv_dim):
        super(Generator, self).__init__()
        linear = []
        linear.append(nn.Linear(z_dim, 64 * conv_dim))
        linear.append(nn.ReLU(inplace=True))

        conv = []
        conv.append(nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=1, bias=False)) # 4 x 4 -> 7 x 7
        conv.append(nn.ReLU(inplace=True))
        conv.append(nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=3, stride=2, padding=1, bias=False))   # 7 x 7-> 13 x 13
        conv.append(nn.ReLU(inplace=True))
        conv.append(nn.ConvTranspose2d(conv_dim, image_dim, kernel_size=4, stride=2, padding=0, bias=False))    # 13 x 13 -> 28 x 28
        conv.append(nn.Sigmoid())

        self.linear = nn.Sequential(*linear)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), 64*4, 4, 4)
        out = self.conv(out)
        return out


class Inverter(nn.Module):
    def __init__(self, z_dim, image_dim, conv_dim):
        super(Inverter, self).__init__()
        conv = []
        conv.append(nn.Conv2d(image_dim, conv_dim, kernel_size=3, stride=2, padding=1))    # 28 x 28 -> 15 x 15
        conv.append(nn.LeakyReLU(inplace=True))
        conv.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1))   # 15 x 15 ->  8 x 8
        conv.append(nn.LeakyReLU(inplace=True))
        conv.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1)) # 8 x 8 -> 4 x 4
        conv.append(nn.LeakyReLU(inplace=True))

        linear = []
        linear.append(nn.Linear(conv_dim*64, 512))
        linear.append(nn.LeakyReLU(inplace=True))
        linear.append(nn.Linear(512, z_dim))

        self.conv = nn.Sequential(*conv)
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 64*64)
        out = self.linear(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, image_dim, conv_dim):
        super(Discriminator, self).__init__()
        conv = []
        conv.append(nn.Conv2d(image_dim, conv_dim, kernel_size=3, stride=2, padding=1))    # 28 x 28 -> 15 x 15
        conv.append(nn.LeakyReLU(inplace=True))
        conv.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1))   # 15 x 15 -> 8 x 8
        conv.append(nn.LeakyReLU(inplace=True))
        conv.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1)) # 8 x 8 -> 4 x 4
        conv.append(nn.LeakyReLU(inplace=True))

        linear = []
        linear.append(nn.Linear(64*64, 1))

        self.conv = nn.Sequential(*conv)
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 64*64)
        out = self.linear(out)
        return out
