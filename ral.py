import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from tensorboardX import  SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from copy import deepcopy


from discriminator import NLayerDiscriminator
from pixelcnn import PixelCNN
from vqvae import VQVAE_2D

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

    
class RAL():
    def __init__(self,
                 opt,
                 epochs,
                 batch_size,
                 gamma,
                 lambda_gp,
                 policy_lr,
                 rf_lr,
                 train_policy_iter,
                 train_rf_iter,
                 pretrain_rf_iters,
                 sample_num,
                 wrtier_path,
                 log_path,
                 ckpt_path
                 ):
        self.args = opt
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_gp = lambda_gp
        self.policy_lr = policy_lr
        self.rf_lr = rf_lr
        self.train_policy_iter = train_policy_iter
        self.train_rf_iter = train_rf_iter
        self.pretrain_rf_iters = pretrain_rf_iters
        self.sample_num = sample_num
        self.eps = 1e-8
        self.writer_path = wrtier_path
        self.log_path = log_path
        self.ckpt_path = ckpt_path

        self.vq = VQVAE_2D( in_channels=opt.vq_in_channels,
                            image_size=opt.vq_image_size,
                            num_hiddens=opt.vq_num_hiddens, 
                            compress_factor=opt.vq_compress_factor,
                            num_residual_layers=opt.vq_num_residual_layers, 
                            num_residual_hiddens=opt.vq_num_residual_hiddens, 
                            num_embeddings=opt.vq_num_embeddings, 
                            embedding_dim=opt.vq_embedding_dim, 
                            commitment_cost=opt.vq_commitment_cost).cuda(opt.device)

        self.G = PixelCNN(  nlayers=opt.pixelcnn_nlayers,
                            in_channels=opt.vq_embedding_dim,
                            nfeats=opt.pixelcnn_nfeats,
                            Klevels=opt.vq_num_embeddings,
                            embedding_dim=opt.vq_embedding_dim).cuda(opt.device)

        self.D = NLayerDiscriminator(   input_nc=opt.vq_in_channels,
                                        ndf=opt.patch_d_ndf,
                                        n_layers=opt.vq_compress_factor).cuda(opt.device)

        self.optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.policy_lr, betas=(0.5, 0.999), eps=1e-8)
        self.optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.rf_lr, betas=(0.5, 0.999), eps=1e-8)

        self.writer = SummaryWriter(self.writer_path)

        # Load pretrained models
        checkpoint = torch.load('results/VQVAE_2D_Com-3-Img-128x128/CKPT/VQVAE.tar', map_location=lambda storage, loc: storage.cuda(opt.device))
        self.vq.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load('results/PixelCNN_VQ-Com-3-Img-128x128/CKPT/PixelCNN.tar', map_location=lambda storage, loc: storage.cuda(opt.device))
        self.G.load_state_dict(checkpoint['model_state_dict'])


    def train(self, train_loader):

        one = torch.tensor(1, dtype=torch.float).cuda(self.args.device)
        mone = one * -1

        step = 0

        f = open(self.log_path, 'a+')

        for p in self.vq.parameters():
            p.requires_grad = False

        # Pretrain discriminator for some iterations
        print('-------------pretrain discriminator-------------')
        for i in tqdm(range(self.pretrain_rf_iters)):
            data = self.get_infinite_batches(train_loader)
            img = data.__next__()

            self.D.zero_grad()
            real_img, _ = self.vq(img)
            latent_codes = self.G.sample([self.args.pixelcnn_image_size, self.args.pixelcnn_image_size], batch_size=self.batch_size, device=img.device)
            latent_codes = latent_codes.reshape(-1, 1)
            inputs = torch.zeros(self.batch_size, self.args.pixelcnn_image_size, self.args.pixelcnn_image_size, self.args.vq_embedding_dim).to(latent_codes.device)
            fake_img = self.vq.decode(latent_codes, inputs)

            # Train with real images
            d_loss_real = self.D(real_img)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(mone)

            # Train with real images
            d_loss_fake = self.D(fake_img)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one)

            # Train with gradient penalty
            gradient_penalty = self.calculate_gradient_penalty(real_img.data, fake_img.data)
            gradient_penalty.backward()

            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            Wasserstein_D = d_loss_real - d_loss_fake
            self.optimizer_D.step()
            

        for epoch in range(self.epochs):
            for data in tqdm(train_loader):
                img = data
                step += 1

                total_d_loss_real = 0
                total_d_loss_fake = 0
                total_Wasserstein_D = 0
                total_g_loss = 0
                total_d_loss = 0
                total_reward = 0
                
                for g_iter in range(self.train_policy_iter):
                    for p in self.D.parameters():
                        p.requires_grad = True
                    
                    d_loss_real = 0
                    d_loss_fake = 0
                    Wasserstein_D = 0

                    # Train discriminator
                    for d_iter in range(self.train_rf_iter):
                        self.D.zero_grad()
                        real_img, _ = self.vq(img)
                        latent_codes = self.G.sample([self.args.pixelcnn_image_size, self.args.pixelcnn_image_size], batch_size=self.batch_size, device=img.device)
                        latent_codes = latent_codes.reshape(-1, 1)
                        inputs = torch.zeros(self.batch_size, self.args.pixelcnn_image_size, self.args.pixelcnn_image_size, self.args.vq_embedding_dim).to(latent_codes.device)
                        fake_img = self.vq.decode(latent_codes, inputs)

                        # Train with real images
                        d_loss_real = self.D(real_img)
                        d_loss_real = d_loss_real.mean()
                        d_loss_real.backward(mone)

                        # Train with real images
                        d_loss_fake = self.D(fake_img)
                        d_loss_fake = d_loss_fake.mean()
                        d_loss_fake.backward(one)

                        # Train with gradient penalty
                        gradient_penalty = self.calculate_gradient_penalty(real_img.data, fake_img.data)
                        gradient_penalty.backward()

                        d_loss = d_loss_fake - d_loss_real + gradient_penalty
                        Wasserstein_D = d_loss_real - d_loss_fake
                        self.optimizer_D.step()
                    
                    # Train Generator
                    for p in self.D.parameters():
                        p.requires_grad = False
                    
                    self.G.zero_grad()

                    # Reinforcement Learning
                    obs_list = []
                    act_list = []
                    lnp_list = []
                    rew_list = []

                    x = torch.zeros(self.batch_size, 1, self.args.pixelcnn_image_size, self.args.pixelcnn_image_size).long().to(img.device)
                    for i in range(x.shape[2]):
                        for j in range(x.shape[3]):
                            obs_list += [deepcopy(x)]
                            logits = self.G.forward(obs_list[-1])
                            probs = torch.softmax(logits[:, :, i, j], 1)
                            m = Categorical(probs)
                            action = m.sample()
                            lnp_list += [m.log_prob(action)]
                            # sample = probs.multinomial(1)
                            x[:, :, i, j] = action
                    
                    latent_codes = x.reshape(-1, 1)
                    inputs = torch.zeros(self.batch_size, self.args.pixelcnn_image_size, self.args.pixelcnn_image_size, self.args.vq_embedding_dim).to(latent_codes.device)
                    fake_img = self.vq.decode(latent_codes, inputs)
                    rewards = self.D(fake_img).reshape(-1)
                    rewards = rewards / torch.max(torch.abs(rewards))
                    
                    R = 0
                    policy_loss = []
                    for r in rewards.tolist()[::-1]:
                        R = r + self.gamma * R
                        rew_list.insert(0, R)
                    returns = torch.tensor(rew_list).to(img.device)
                    # returns = (returns - returns.mean()) / (returns.std() + self.eps)
                    for log_prob, R in zip(lnp_list, returns):
                        policy_loss.append(-log_prob * R)
                    
                    g_loss = sum(policy_loss)
                    g_loss.backward()
                    self.optimizer_G.step()

                    self.writer.add_scalar('RAL_Train/Wasserstein distance', Wasserstein_D.item(), global_step=step)
                    self.writer.add_scalar('RAL_Train/Loss D', d_loss.item(), global_step=step)
                    self.writer.add_scalar('RAL_Train/Loss G', g_loss.item(), global_step=step)
                    self.writer.add_scalar('RAL_Train/Loss D Real', d_loss_real.item(), global_step=step)
                    self.writer.add_scalar('RAL_Train/Loss D Fake', d_loss_fake.item(), global_step=step)
                    self.writer.add_scalar('RAL_Train/Reward', rewards.mean().item(), global_step=step)

                    total_Wasserstein_D += Wasserstein_D.item()
                    total_d_loss += d_loss.item()
                    total_g_loss += g_loss.item()
                    total_d_loss_real += d_loss_real.item()
                    total_d_loss_fake += d_loss_fake.item()
                    total_reward += rewards.mean().item()
                    
                if step % 1 == 0:
                    nrow = self.sample_num
                    with torch.no_grad():
                        bs = nrow * 2
                        latent_codes = self.G.sample([self.args.pixelcnn_image_size, self.args.pixelcnn_image_size], batch_size=bs, device=img.device)
                        latent_codes = latent_codes.reshape(-1, 1)
                        inputs = torch.zeros(bs, self.args.pixelcnn_image_size, self.args.pixelcnn_image_size, self.args.vq_embedding_dim).to(latent_codes.device)
                        display_imgs = self.vq.decode(latent_codes, inputs)

                    grid_recon = make_grid(display_imgs.cpu(), nrow=nrow, range=(0, 1), normalize=True)
                    self.writer.add_image('RAL Eval/sample_imgs', grid_recon, step)
            
            f.write("Epoch[%d] W_D: %.4lf Loss_D: %.4lf Loss_G: %.4lf \
                    Loss_D_Real: %.4lf Loss_D_fake: %.4lf Reward: %.4lf\n" \
                    %(epoch, total_Wasserstein_D / len(train_loader), 
                    total_d_loss / len(train_loader),
                    total_g_loss / len(train_loader),
                    total_d_loss_real / len(train_loader),
                    total_d_loss_fake / len(train_loader),
                    total_reward / len(train_loader)))
            
            # save checkpoint
            save_checkpoint(self.G, self.optimizer_G, epoch, self.ckpt_path + 'G_PixelCNN.tar')
            save_checkpoint(self.D, self.optimizer_D, epoch, self.ckpt_path + 'D_PatchGAN.tar')

            # nrow = self.sample_num
            # with torch.no_grad():
            #     bs = nrow * 2
            #     latent_codes = self.G.sample([opt.pixelcnn_image_size, opt.pixelcnn_image_size], batch_size=bs, device=device)
            #     latent_codes = latent_codes.reshape(-1, 1)
            #     inputs = torch.zeros(bs, opt.pixelcnn_image_size, opt.pixelcnn_image_size, opt.vq_embedding_dim).to(latent_codes.device)
            #     display_imgs = self.vq.decode(latent_codes, inputs)

            # grid_recon = make_grid(display_imgs.cpu(), nrow=nrow, range=(0, 1), normalize=True)
            # writer.add_image('Eval/sample_imgs', grid_recon, epoch)

        f.close()

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(real_images.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(eta.device),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

        return grad_penalty

    def get_infinite_batches(self, data_loader):
        while True:
            for i, images in enumerate(data_loader):
                yield images

    def load_model(self, checkpoint_path):
        G_model_path = checkpoint_path + 'G_PixelCNN.tar'
        D_model_path = checkpoint_path + 'D_PatchGAN.tar'
        G_checkpoint = torch.load(G_model_path, map_location=lambda storage, loc: storage.cuda(self.args.device))
        D_checkpoint = torch.load(D_model_path, map_location=lambda storage, loc: storage.cuda(self.args.device))
        self.G.load_state_dict(G_checkpoint['model_state_dict'])
        self.D.load_state_dict(D_checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(G_checkpoint['optimizer_state_dict'])
        self.optimizer_D.load_state_dict(D_checkpoint['optimizer_state_dict'])