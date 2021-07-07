import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchkeras
import tensorboardX
from tensorboardX import  SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision.utils import make_grid
import scheduler

import numpy as np
from tqdm import tqdm
from sys import path
import time
import os

import sys
sys.path.append('/home/vision/diska2/1JYK/RL/Project')
from vqvae import VQVAE_2D
from pixelcnn import PixelCNN
from ral import RAL
from datas import CelebA, save_image_tensor
from parameters import gen_config, save_config


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return False
    else:
        return True

def save_model(model, path):
    torch.save(model, path)

def load_model(path, cuda_device):
    if torch.cuda.is_available():
        model = torch.load(path, map_location=lambda storage, loc: storage.cuda(cuda_device))
    else:
        model = torch.load(path)

    return model

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path, cuda_device=0):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda(cuda_device)
    return model, optimizer, epoch

def merge_images(images, fixed_imgs, frame_raw):
    images = torch.split(images, frame_raw, dim=0)
    fixed_imgs = torch.split(fixed_imgs, frame_raw, dim=0)
    merged_images = torch.cat([torch.cat([image, fixed_img], dim=0) for image, fixed_img in zip(images, fixed_imgs)], dim=0)
    return merged_images


def train_VQ_2D(opt, checkpoint_path=None):
    # model_name = 'VQVAE_2D_' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    # model_name = 'VQVAE_Emb-{}x{}/'.format(opt.vq_num_embeddings, opt.vq_embedding_dim)
    model_name = 'VQVAE_2D_Com-{}-Img-{}x{}/'.format(opt.vq_compress_factor, opt.vq_image_size, opt.vq_image_size)
    folder_path = 'results/' + model_name
    writer_path = folder_path + 'SummaryWriter/'
    ckpt_path = folder_path + 'CKPT/'
    config_path = folder_path + 'config.json'
    log_path = folder_path + 'log.txt'
    exists_or_mkdir(folder_path)
    exists_or_mkdir(writer_path)
    exists_or_mkdir(ckpt_path)
    save_config(opt, config_path)

    train_loader, valid_loader = CelebA(opt.vq_batch_size, opt.device)

    current_step = 0
    epoch_current = 0

    # model = VQVAE(  in_channels=opt.vq_in_channels, 
    #                 num_embeddings=opt.vq_num_embeddings, 
    #                 embedding_size=opt.vq_embedding_dim, 
    #                 res_hidden_channels=opt.vq_num_residual_layers, 
    #                 commitment_cost=opt.vq_commitment_cost).cuda(opt.device)

    model = VQVAE_2D(  in_channels=opt.vq_in_channels,
                       image_size=opt.vq_image_size,
                       num_hiddens=opt.vq_num_hiddens,
                       compress_factor=opt.vq_compress_factor,
                       num_residual_layers=opt.vq_num_residual_layers,
                       num_residual_hiddens=opt.vq_num_residual_hiddens,
                       num_embeddings=opt.vq_num_embeddings,
                       embedding_dim=opt.vq_embedding_dim,
                       commitment_cost=opt.vq_commitment_cost).cuda(opt.device)

    device = next(model.parameters()).device

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.vq_lr, eps=1e-8)

    # lr_sch = StepLR(optimizer, 20, gamma=0.5)
    # sch_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=lr_sch)
    
    # lr_sch = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    # sch_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=lr_sch)

    # optimizer.step()
    # sch_warmup.step()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model, optimizer, epoch_current, loss = load_checkpoint(model, optimizer, checkpoint_path)
        print("Pre-trained models loaded.")
    
    # for epoch in range(epoch_current, opt.epochs):
    #     print(optimizer.param_groups[0]['lr'])
    #     sch_warmup.step()

    writer = SummaryWriter(writer_path)
    f = open(log_path, 'a+')

    best_valid_loss = np.inf
    step = 0
    for epoch in range(epoch_current, opt.vq_epochs):
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        total_train_loss = 0
        
        for data in tqdm(train_loader):
            model.zero_grad()
            
            img = data
            x_recon, vq_loss = model(img)
            recon_loss = nn.MSELoss()(img, x_recon)
            loss = recon_loss + vq_loss
            total_train_loss += recon_loss.item()
            # print(recon_loss.item(), vq_loss.item())
            writer.add_scalar('VQ Train/step_recon_loss', recon_loss.item(), global_step=step)
            writer.add_scalar('VQ Train/step_vq_loss', vq_loss.item(), global_step=step)

            loss.backward()
            optimizer.step()
            step += 1

        # sch_warmup.step()
        print("Epoch[%d] train loss: %.5lf" %(epoch, total_train_loss / len(train_loader)))
        writer.add_scalar('VQ Train/epoch_recon_loss', total_train_loss / len(train_loader), global_step=epoch)
        
        total_valid_loss = 0
        with torch.no_grad():
            for data in tqdm(valid_loader):
                img = data
                x_recon, vq_loss = model(img)
                recon_loss = nn.MSELoss()(img, x_recon)
                total_valid_loss += recon_loss.item()
        
        print("Epoch[%d] valid loss: %.5lf" %(epoch, total_valid_loss / len(valid_loader)))
        writer.add_scalar('VQ Eval/epoch_recon_loss', total_valid_loss / len(valid_loader), global_step=epoch)

        f.write("Epoch[%d] train loss: %.5lf valid loss: %.5lf\n" %(epoch, total_train_loss / len(train_loader), total_valid_loss / len(valid_loader)))

        nrow = opt.vq_recon_sample_num
        with torch.no_grad():
            for data in tqdm(valid_loader):
                raw_img = data
                recon_img, _ = model(raw_img)
                break
                
        display_imgs = merge_images(raw_img[:nrow], recon_img[:nrow], nrow)
        grid_recon = make_grid(display_imgs.cpu(), nrow=nrow, range=(0, 1), normalize=True)
        writer.add_image('VQ Eval/reconstruction', grid_recon, epoch)

        save_checkpoint(model, optimizer, epoch, ckpt_path + 'VQVAE.tar')

        if total_valid_loss / len(valid_loader) < best_valid_loss:
            best_valid_loss = total_valid_loss / len(valid_loader)
            save_model(model, ckpt_path + 'VQVAE.pth')

    f.close()


def train_pixelcnn(opt, checkpoint_path=None):
    # vq_ckpt_path = 'results/VQVAE_Emb-{}x{}/CKPT/VQVAE.tar'.format(opt.vq_num_embeddings, opt.vq_embedding_dim)
    vq_ckpt_path = 'results/VQVAE_2D_Com-{}-Img-{}x{}/CKPT/VQVAE.tar'.format(opt.vq_compress_factor, opt.vq_image_size, opt.vq_image_size)
    model_name = 'PixelCNN_VQ-Com-{}-Img-{}x{}/'.format(opt.vq_compress_factor, opt.vq_image_size, opt.vq_image_size)
    folder_path = 'results/' + model_name
    writer_path = folder_path + 'SummaryWriter/'
    ckpt_path = folder_path + 'CKPT/'
    config_path = folder_path + 'config.json'
    log_path = folder_path + 'log.txt'
    exists_or_mkdir(folder_path)
    exists_or_mkdir(writer_path)
    exists_or_mkdir(ckpt_path)
    save_config(opt, config_path)

    train_loader, valid_loader = CelebA(opt.pixelcnn_batch_size, opt.device)

    current_step = 0
    epoch_current = 0

    # vq = VQVAE( in_channels=opt.vq_in_channels, 
    #             num_embeddings=opt.vq_num_embeddings, 
    #             embedding_size=opt.vq_embedding_dim, 
    #             res_hidden_channels=vq_num_residual_hiddens, 
    #             commitment_cost=opt.vq_commitment_cost).cuda(opt.device)

    vq = VQVAE_2D(  in_channels=opt.vq_in_channels,
                    image_size=opt.vq_image_size,
                    num_hiddens=opt.vq_num_hiddens,
                    compress_factor=opt.vq_compress_factor,
                    num_residual_layers=opt.vq_num_residual_layers,
                    num_residual_hiddens=opt.vq_num_residual_hiddens,
                    num_embeddings=opt.vq_num_embeddings,
                    embedding_dim=opt.vq_embedding_dim,
                    commitment_cost=opt.vq_commitment_cost).cuda(opt.device)
    
    checkpoint = torch.load(vq_ckpt_path, map_location=lambda storage, loc: storage.cuda(opt.device))
    vq.load_state_dict(checkpoint['model_state_dict'])

    model = PixelCNN(nlayers=opt.pixelcnn_nlayers,
                     in_channels=opt.vq_embedding_dim,
                     nfeats=opt.pixelcnn_nfeats,
                     Klevels=opt.vq_num_embeddings,
                     embedding_dim=opt.vq_embedding_dim).cuda(opt.device)

    # model = GatedPixelCNN(  in_channels=opt.vq_num_embeddings, 
    #                         hidden_channels=opt.pixelcnn_hidden_channels, 
    #                         output_channels=opt.pixelcnn_hidden_channels, 
    #                         num_layers=opt.pixelcnn_nlayers)

    device = next(model.parameters()).device

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.pixelcnn_lr, eps=1e-8)

    lr_sch = StepLR(optimizer, 20, gamma=0.9)
    sch_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=lr_sch)
    
    # lr_sch = CosineAnnealingLR(optimizer, T_max=5, eta_min=2e-4)
    # sch_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=lr_sch)

    optimizer.step()
    sch_warmup.step()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model, optimizer, epoch_current, loss = load_checkpoint(model, optimizer, checkpoint_path)
        print("Pre-trained models loaded.")
    
    # for epoch in range(epoch_current, opt.pixelcnn_epochs):
    #     print(optimizer.param_groups[0]['lr'])
    #     sch_warmup.step()
    
    writer = SummaryWriter(writer_path)
    f = open(log_path, 'a+')

    best_valid_loss = np.inf
    step = 0
    for epoch in range(epoch_current, opt.pixelcnn_epochs):
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        total_train_loss = 0
        total_train_accu = 0
        
        for data in tqdm(train_loader):
            img = data
            with torch.no_grad():
                encoding_indices, inputs = vq.encode(img)
            x = encoding_indices.reshape(opt.pixelcnn_batch_size, opt.pixelcnn_in_channels, opt.pixelcnn_image_size, opt.pixelcnn_image_size)
            
            model.zero_grad()
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, x.squeeze(1))
            pred_x = torch.max(logits, dim=1, keepdim=True)[1]
            accuracy = torch.sum(x == pred_x) / torch.numel(x)

            writer.add_scalar('PixelCNN Train/step_crossentropy_loss', loss.item(), global_step=step)
            writer.add_scalar('PixelCNN Train/step_accuracy_loss', accuracy.item(), global_step=step)

            loss.backward()
            optimizer.step()
            step += 1
            total_train_loss += loss.item()
            total_train_accu += accuracy.item()

        # sch_warmup.step()
        print("Epoch[%d] train loss: %.5lf" %(epoch, total_train_loss / len(train_loader)))
        writer.add_scalar('PixelCNN Train/epoch_crossentropy_loss', total_train_loss / len(train_loader), global_step=epoch)
        writer.add_scalar('PixelCNN Train/epoch_accuracy', total_train_accu / len(train_loader), global_step=epoch)
        
        total_valid_loss = 0
        total_valid_accu = 0
        with torch.no_grad():
            model.eval()
            for data in tqdm(valid_loader):
                img = data
                encoding_indices, inputs = vq.encode(img)
                x = encoding_indices.reshape(img.shape[0], opt.pixelcnn_in_channels, opt.pixelcnn_image_size, opt.pixelcnn_image_size)
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits, x.squeeze(1))
                pred_x = torch.max(logits, dim=1, keepdim=True)[1]
                accuracy = torch.sum(x == pred_x) / torch.numel(x)

                total_valid_loss += loss.item()
                total_valid_accu += accuracy.item()
        
        print("Epoch[%d] valid loss: %.5lf" %(epoch, total_valid_loss / len(valid_loader)))
        writer.add_scalar('PixelCNN Eval/epoch_crossentropy_loss', total_valid_loss / len(valid_loader), global_step=epoch)
        writer.add_scalar('PixelCNN Eval/epoch_accuracy', total_valid_accu / len(valid_loader), global_step=epoch)

        f.write("Epoch[%d] train loss: %.5lf valid loss: %.5lf\n" %(epoch, total_train_loss / len(train_loader), total_valid_loss / len(valid_loader)))

        nrow = opt.pixelcnn_recon_sample_num
        with torch.no_grad():
            bs = nrow * 2
            latent_codes = model.sample([opt.pixelcnn_image_size, opt.pixelcnn_image_size], batch_size=bs, device=device)
            latent_codes = latent_codes.reshape(-1, 1)
            inputs = torch.zeros(bs, opt.pixelcnn_image_size, opt.pixelcnn_image_size, opt.vq_embedding_dim).to(latent_codes.device)
            display_imgs = vq.decode(latent_codes, inputs)

        grid_recon = make_grid(display_imgs.cpu(), nrow=nrow, range=(0, 1), normalize=True)
        writer.add_image('PixelCNN Eval/sample_imgs', grid_recon, epoch)

        save_checkpoint(model, optimizer, epoch, ckpt_path + 'PixelCNN.tar')

        if total_valid_loss / len(valid_loader) < best_valid_loss:
            best_valid_loss = total_valid_loss / len(valid_loader)
            save_model(model, ckpt_path + 'PixelCNN.pth')

    f.close()
    


def train_ral(opt, checkpoint_path=None):
    model_name = 'RAL/'
    folder_path = 'results/' + model_name
    writer_path = folder_path + 'SummaryWriter/'
    ckpt_path = folder_path + 'CKPT/'
    config_path = folder_path + 'config.json'
    log_path = folder_path + 'log.txt'
    exists_or_mkdir(folder_path)
    exists_or_mkdir(writer_path)
    exists_or_mkdir(ckpt_path)
    save_config(opt, config_path)

    train_loader, valid_loader = CelebA(opt.ral_batch_size, opt.device)

    model = RAL( opt=opt,
                 epochs=opt.ral_epochs,
                 batch_size=opt.ral_batch_size,
                 gamma=opt.ral_gamma,
                 lambda_gp=opt.ral_lambda_gp,
                 policy_lr=opt.ral_policy_lr,
                 rf_lr=opt.ral_rf_lr,
                 train_policy_iter=opt.ral_train_policy_iter,
                 train_rf_iter=opt.ral_train_rf_iter,
                 pretrain_rf_iters=opt.ral_pretrain_rf_iters,
                 sample_num=opt.ral_sample_num,
                 wrtier_path=writer_path,
                 log_path=log_path,
                 ckpt_path=ckpt_path)

    model.train(train_loader)



if __name__ == '__main__':
    opt = gen_config()

    # train_VQ_2D(opt)
    # train_pixelcnn(opt)
    train_ral(opt)