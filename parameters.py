import os
import json
import argparse


def gen_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=2)
    
    parser.add_argument('--vq_epochs', type=int, default=100)
    parser.add_argument('--vq_lr', type=float, default=2e-4)
    parser.add_argument('--vq_batch_size', type=int, default=32)
    parser.add_argument('--vq_in_channels', type=int, default=3)
    parser.add_argument('--vq_image_size', type=int, default=128)
    parser.add_argument('--vq_num_hiddens', type=int, default=256)
    parser.add_argument('--vq_compress_factor', type=int, default=3)
    parser.add_argument('--vq_num_residual_layers', type=int, default=3)
    parser.add_argument('--vq_num_residual_hiddens', type=int, default=64)
    parser.add_argument('--vq_num_embeddings', type=int, default=256)
    parser.add_argument('--vq_embedding_dim', type=int, default=256)
    parser.add_argument('--vq_commitment_cost', type=float, default=0.25)
    parser.add_argument('--vq_recon_sample_num', type=int, default=6)

    parser.add_argument('--pixelcnn_epochs', type=int, default=100)
    parser.add_argument('--pixelcnn_lr', type=float, default=1e-3)
    parser.add_argument('--pixelcnn_batch_size', type=int, default=32)
    parser.add_argument('--pixelcnn_in_channels', type=int, default=1)
    parser.add_argument('--pixelcnn_image_size', type=int, default=16)
    parser.add_argument('--pixelcnn_nlayers', type=int, default=16)
    parser.add_argument('--pixelcnn_nfeats', type=int, default=128)
    parser.add_argument('--pixelcnn_Klevels', type=int, default=256)
    parser.add_argument('--pixelcnn_hidden_channels', type=int, default=256)
    parser.add_argument('--pixelcnn_recon_sample_num', type=int, default=6)

    parser.add_argument('--patch_d_ndf', type=int, default=64)
    parser.add_argument('--ral_epochs', type=int, default=100)
    parser.add_argument('--ral_batch_size', type=int, default=1)
    parser.add_argument('--ral_gamma', type=float, default=0.9)
    parser.add_argument('--ral_lambda_gp', type=float, default=10.0)
    parser.add_argument('--ral_policy_lr', type=float, default=1e-4)
    parser.add_argument('--ral_rf_lr', type=float, default=4e-6)
    parser.add_argument('--ral_train_policy_iter', type=int, default=1)
    parser.add_argument('--ral_train_rf_iter', type=int, default=5)
    parser.add_argument('--ral_pretrain_rf_iters', type=int, default=100)
    parser.add_argument('--ral_sample_num', type=int, default=6)

    args = parser.parse_args()

    return args


def save_config(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_config(load_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(load_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


if __name__ == '__main__':
    parser = gen_config()
    print(parser.epochs)
    # save_config(parser, 'config.json')
    # config = load_config('config.json')
