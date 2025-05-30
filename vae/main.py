from __future__ import print_function
import argparse
import copy
import os
import sys
import shutil
import torch
import pandas as pd
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from adamw_overshoot_delayed import AdamO


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--results_dir', type=str, default="/home/kopal/benchmarking-overshoot/results")
parser.add_argument('--job_name', type=str, required=True)
parser.add_argument('--optimizer_name', type=str, required=True)
parser.add_argument('--overshoot', type=float, default=4.0)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--accel', action='store_true', 
                    help='use accelerator')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


torch.manual_seed(args.seed)

device = torch.device("cuda")
print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
if args.optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
elif args.optimizer_name == "adamo":
    optimizer = AdamO(model.parameters(), lr=1e-3, weight_decay=0, overshoot=args.overshoot)
else:
    raise Exception(f"Unknown optimizer: {args.optimizer_name}")


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss, shifted_train_loss = 0, 0
    train_stats = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        
        if hasattr(optimizer, 'move_to_base'):
            optimizer.move_to_base()
            eval_model = copy.deepcopy(model)
            optimizer.move_to_overshoot()
            with torch.no_grad():    
                recon_batch, mu, logvar = eval_model(data)
                shifted_train_loss += loss_function(recon_batch, data, mu, logvar).item()
        
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
                
            iteration = batch_idx + epoch * len(train_loader)
            log_writer.add_scalar("train_basic_loss", loss.item(), iteration)
            train_stats.append({"train_basic_loss": loss.item()})
            if hasattr(optimizer, 'move_to_base'):
                optimizer.move_to_base()
                eval_model = copy.deepcopy(model)
                optimizer.move_to_overshoot()
                with torch.no_grad():    
                    recon_batch, mu, logvar = model(data)
                    shifted_loss = loss_function(recon_batch, data, mu, logvar)
                log_writer.add_scalar("train_shifted_to_base_loss", shifted_loss, iteration)
                train_stats[-1]["train_shifted_to_base_loss"] = shifted_loss
            else:
                log_writer.add_scalar("train_shifted_to_base_loss", loss.item(), iteration)
                train_stats[-1]["train_shifted_to_base_loss"] = loss.item()
                
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    return train_stats


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

if __name__ == "__main__":
    
    # Prepare logs
    base_dir = os.path.join(args.results_dir, "vae", f"{args.job_name}_{args.optimizer_name}_{args.overshoot}")
    os.makedirs(base_dir, exist_ok=True)
    # version_dir = os.path.join(base_dir, f"version_{len(os.listdir(base_dir)) + 1}")
    version_dir = os.path.join(base_dir, f"seed_{args.seed}")
    log_writer = SummaryWriter(log_dir=version_dir) # type: ignore
    shutil.copy(sys.argv[0], os.path.join(version_dir, '__'.join(sys.argv)))
    
    train_stats, test_stats = [], []
    for epoch in range(1, args.epochs + 1):
        train_stats.extend(train(epoch))
        if isinstance(optimizer, AdamO):
            optimizer.move_to_base()
        test_loss = test(epoch)
        log_writer.add_scalar("test_loss", test_loss, epoch)
        test_stats.append({"test_loss": test_loss})
        if isinstance(optimizer, AdamO):
            optimizer.move_to_overshoot()
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

    pd.DataFrame(train_stats).to_csv(os.path.join(version_dir, "training_stats.csv"), index=False)
    pd.DataFrame(test_stats).to_csv(os.path.join(version_dir, "validation_stats.csv"), index=False)
