from torch import nn
from torch.autograd import Variable
from torch import optim
import torch
import torchvision.utils as tvu
#from tensorboardX import SummaryWriter
import shutil
from tqdm import tqdm
import numpy as np

from torch import nn
import torch


def undo_transform(x_in, means = (0.5070, 0.4865, 0.4409), vars = (0.2673, 0.2564, 0.2761)):
    means_t = torch.tensor(means).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    var_t = torch.tensor(vars).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return (x_in*var_t+means_t)


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)

        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)

        # Decoder
        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, 3, 32, 32)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Trainer:
    def __init__(self, model, loss, train_loader, test_loader, args):
        self.model = model
        self.args = args
        self.args.start_epoch = 0

        self.train_loader = train_loader
        self.test_loader = test_loader

        # Loss function and Optimizer
        self.loss = loss
        self.optimizer = self.get_optimizer()

        # Tensorboard Writer
        #self.summary_writer = SummaryWriter(log_dir=args.summary_dir)
        # Model Loading
        if args.resume:
            self.load_checkpoint(self.args.resume_from)

    def train(self):
        self.model.train()
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
                if self.args.cuda:
                    data = data.cuda()
                data = Variable(data)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss(recon_batch, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.cpu().item())

            print("epoch {}: - loss: {}".format(epoch, np.mean(np.array(loss_list))))
            new_lr = self.adjust_learning_rate(epoch)
            print('learning rate:', new_lr)

            #self.summary_writer.add_scalar('training/loss', np.mean(loss_list), epoch)
            #self.summary_writer.add_scalar('training/learning_rate', new_lr, epoch)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            })
            #if epoch % self.args.test_every == 0:
            self.test(epoch)

            
    def test(self, cur_epoch):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.test_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss(recon_batch, data, mu, logvar)
            #indices.data = indices.data.float() / 255
            if i==0:
                #print(recon_batch.shape, data.shape)
                n = min(data.size(0), 16)
                comparison = undo_transform(torch.cat([data[:n], recon_batch.view(-1, 3, 32, 32)[:n]], dim=0).cpu())
                tvu.save_image(comparison, 'results/reconstruction_vae_' + str(cur_epoch) + '.png', nrow=n)

                print("Drawing some samples")
                with torch.no_grad():
                    # Sample 10*nSamplesEach random latent vectors
                    z_sample = torch.randn(self.args.n_samples, 512)
                    # Assign them labels from 0...9
                    x_sample = self.model.decode(z_sample.cuda())
                    tvu.save_image(undo_transform(x_sample.cpu()), 'results/vaesample_' + str(cur_epoch) + '.png', nrow=10)
            #if i == 0:
            #    n = min(data.size(0), 8)
            #    comparison = torch.cat([data[:n],
            #                            indices.view(-1, 3, 32, 32)[:n]])
            #    #self.summary_writer.add_image('testing_set/image', comparison, cur_epoch)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def test_on_trainings_set(self):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.train_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss(recon_batch, data, mu, logvar).data[0]
            _#, indices = recon_batch.max(1)
            i#ndices.data = indices.data.float() / 255
            if i % 50 == 0:
                n = min(data.size(0), 16)
                comparison = undo_transform(torch.cat([data[:n], recon_batch.view(-1, 3, 32, 32)[:n]], dim=0))
                #tvu.save_image(comparison, 'results/reconstruction_wae_' + str(epoch) + '.png', nrow=n)
                #self.summary_writer.add_image('training_set/image', comparison, i)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test on training set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        return learning_rate

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        '''
        a function to save checkpoint of the training
        :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        :param is_best: boolean to save the checkpoint aside if it has the best score so far
        :param filename: the name of the saved file
        '''
        torch.save(state, self.args.checkpoint_dir + filename)
        if is_best:
            shutil.copyfile(self.args.checkpoint_dir + filename,
                            self.args.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.checkpoint_dir))
