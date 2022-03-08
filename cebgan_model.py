import torch
import torch.nn as nn
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description='TrainToDriveTest')
parser.add_argument('--conv_bias', type=bool, default=True)
parser.add_argument('--linear_bias', type=bool, default=True)
parser.add_argument('--cuda', type =bool, default =True)
parser.add_argument('--zSize', type=int, default=10)
parser.add_argument('--nZ', type =int, default =32)
parser.add_argument('--nG', type =int, default =4)
parser.add_argument('--oSize', type =int, default=1)
parser.add_argument('--drop_rate', type =float, default=0.0)
parser.add_argument('--imgH', type =int, default= 128)
parser.add_argument('--imgW', type =int, default= 128)
parser.add_argument('--imgCh', type =int, default= 3)
parser.add_argument('--resumeG_from', type =str, default='/opt/carla-simulator/PythonAPI/examples/new-models/netG_ep101.pth') # exp16 - ep41
parser.add_argument('--resumeD_from', type =str, default='/opt/carla-simulator/PythonAPI/examples/new-models/netD_ep101.pth')
args = parser.parse_args()



def weights_init_gen(m):
    class_name = m.__class__.__name__
    std = 0.05
    if class_name.find('Conv') != -1:
        m.weight.data.copy_(torch.randn(m.weight.data.size()).mul_(std))
        if args.conv_bias:
            m.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        m.weight.data.copy_(torch.randn(m.weight.data.size()).mul_(std))
        if args.linear_bias:
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.copy_(torch.randn(m.weight.data.size()).mul_(std))


def weights_init_disc(m):
    class_name = m.__class__.__name__
    std = 0.05
    if class_name.find('Conv') != -1:
        m.weight.data.copy_(torch.randn(m.weight.data.size()).mul_(std))
        if args.conv_bias:
            m.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        m.weight.data.copy_(torch.randn(m.weight.data.size()).mul_(std))
        if args.linear_bias:
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.copy_(torch.randn(m.weight.data.size()).mul_(std))


img_fe = nn.Sequential(
    nn.BatchNorm2d(3),
    nn.Conv2d(3, args.nG, 4, 2, 1, bias=args.conv_bias),
    nn.LeakyReLU(0.2, True),
    nn.Dropout(args.drop_rate),
    nn.BatchNorm2d(args.nG),
    nn.Conv2d(args.nG, 2 * args.nG, 4, 2, 1, bias=args.conv_bias),
    nn.LeakyReLU(0.2, True),
    nn.Dropout(args.drop_rate),
    nn.BatchNorm2d(2 * args.nG),
    nn.Conv2d(2 * args.nG, 4 * args.nG, 4, 2, 1, bias=args.conv_bias),
    nn.LeakyReLU(0.2, True),
    nn.Dropout(args.drop_rate),
    nn.BatchNorm2d(4 * args.nG),
    nn.Conv2d(4 * args.nG, 4 * args.nG, 4, 2, 1, bias=args.conv_bias),
    nn.LeakyReLU(0.2, True),
    nn.Dropout(args.drop_rate),
    nn.BatchNorm2d(4 * args.nG),
    nn.Conv2d(4 * args.nG, 8 * args.nG, 4, 2, 1, bias=args.conv_bias),
    nn.LeakyReLU(0.2, True),
    nn.Dropout(args.drop_rate),
    nn.BatchNorm2d(8 * args.nG),
    nn.Conv2d(8 * args.nG, 8 * args.nG, 4, bias=args.conv_bias),
    nn.LeakyReLU(0.2, True),
    nn.Dropout(args.drop_rate))

fc1 = nn.Sequential(
    #nn.BatchNorm1d(args.nZ),
    nn.Linear(args.nZ, args.zSize, bias=args.linear_bias),
    nn.LeakyReLU(0.2, True),
    nn.Dropout(args.drop_rate),
    #nn.BatchNorm1d(args.zSize),
    nn.Linear(args.zSize, args.oSize, bias=args.linear_bias),
    nn.Tanh())


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.noise_fe = nn.Sequential(
            nn.Linear(args.zSize, args.nZ, bias=args.linear_bias),
            nn.LeakyReLU(0.2, True))

        self.img_fe = img_fe
        self.fc1 = fc1

    def forward(self, latent_input, cond_img):
        x0_temp = self.noise_fe(latent_input)
        x1_temp = self.img_fe(cond_img)
        x1_temp = x1_temp.view(-1, args.nZ)
        x = x0_temp + x1_temp
        output = self.fc1(x)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.out_fe = nn.Sequential(
            nn.Linear(args.oSize, args.nZ, bias=args.linear_bias),
            nn.LeakyReLU(0.2, True))


        self.img_fe = img_fe
        self.fc1 = fc1

    def forward(self, traj, cond_img):
        x0_temp = self.out_fe(traj)
        x1_temp = self.img_fe(cond_img)
        x1_temp = x1_temp.view(-1, args.nZ)
        x0 = traj
        x = x1_temp + x0_temp
        reconstruct = self.fc1(x)
        x = reconstruct - x0
        x = x ** 2
        x = torch.sum(x, dim = 1)
        return x

netG = Generator()
#netG.apply(weights_init_gen)

netD = Discriminator()
#netD.apply(weights_init_disc)

if args.cuda:
    netG.cuda()
    netD.cuda()

load_state_G = torch.load(args.resumeG_from)
netG.load_state_dict(load_state_G)
netG.eval()

load_state_D = torch.load(args.resumeD_from)
netD.load_state_dict(load_state_D)
netD.eval()
