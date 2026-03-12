# scheduler.py
import torch

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, device="cuda"):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        self.betas = torch.linspace(0.0001, 0.02, num_train_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise