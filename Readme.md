# A toy Implementation of 《Incorporating Reinforced Adversarial Learning in Autoregressive Image Generation》

Only implementate a simple version of the model (VQVAE + PixelCNN + PatchWGAN + Policy Gradient) proposed in paper 《Incorporating Reinforced Adversarial Learning in Autoregressive Image Generation》, including:

- no VQVAE-2, only VQVAE
- no partial generation
- no single reward, only use patchwgan to provide intermediate rewards
- no oracle PixelCNN

