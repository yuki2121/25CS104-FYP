import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        device = t.device
        t=t.float()
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_dim, device=device).float() / max(half_dim - 1, 1))
        args = t[:, None] * freqs[None]

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1), mode='constant')

        return embedding


class Denoiser(nn.Module):
    # denoiser, predict noise with noisy 3d pose, time step etc

    def __init__(self, joint_num=18, time_embedding_dim=128, hidden_dim=512, depth=6, dropout=0.1):
        super(Denoiser, self).__init__()
        self.joint_num = joint_num
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        input_dim = 3*joint_num+2*joint_num+1*joint_num+1*joint_num+1*joint_num+hidden_dim  # 3D pose + 2D pose + scores + mask + root_type + time embedding 

        layers = []
        dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, joint_num*3)  

    def forward(self, x_t, x2d, scores, mask, t, root_type):
        B, J, _ = x_t.shape
        assert J == self.joint_num

        s= scores.unsqueeze(-1)  # (B, J, 1)
        m = mask.unsqueeze(-1)  # (B, J, 1)
        r = root_type[:, None, None].float().expand(B, J, 1)  # (B, J, 1)

        pose_feat = torch.cat([x_t, x2d, s, m, r], dim=-1) # (B, J, 3+2+1+1+1)
        pose_feat = pose_feat.reshape(B, -1)

        t_emb = self.t_mlp(self.time_embedding(t))  # (B, hidden_dim)

        x= torch.cat([pose_feat, t_emb], dim=-1)  # (B, J, input_dim)
        x = self.backbone(x)  # (B, J, hidden_dim)
        noise_pred = self.output_layer(x)  # (B, J, 3)
        noise_pred = noise_pred.view(B, J, 3)
        return noise_pred


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super(GaussianDiffusion, self).__init__()
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        # forward diffusion: add noise to x0 at time t

        if noise is None:
            noise = torch.randn_like(x0)

        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)  
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # (B, 1, 1)

        return s1 * x0 + s2 * noise, noise

    def predict_x0(self, xt, t, noise_pred):
        # given noisy xt and predicted noise, recover x0
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)  
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # (B, 1, 1)

        return (xt - s2 * noise_pred) / s1.clamp(min=1e-6)

    def p_sample(self, xt, t, noise_pred):
        # backward diffusion: given xt and predicted noise, sample x_{t-1}

        alpha_t = self.alphas[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (xt - ((1.0 - alpha_t) / sqrt_one_minus_alphas_cumprod_t) * noise_pred)
        
        noise = torch.randn_like(xt)

        nonzero_mask = (t > 0).float().view(-1, 1, 1)
        
        x_t_minus_1 = mean + nonzero_mask * torch.sqrt(beta_t) * noise
        
        return x_t_minus_1




