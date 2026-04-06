import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator2D(nn.Module):
    def __init__(self, joint_num=18, hidden_dim=512, use_mask=True, use_scores=True):
        super(Discriminator2D, self).__init__()
        self.joint_num = joint_num
        self.input_dim = joint_num * 2  # 2D keypoints
        if use_mask:
            self.input_dim += joint_num  
        if use_scores:
            self.input_dim += joint_num
        
        self.use_mask = use_mask
        self.use_scores = use_scores

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        

    def forward(self, keypoints_2d, mask = None, scores = None):
        B,J,_ = keypoints_2d.shape  
        assert J == self.joint_num

        feat = [keypoints_2d.reshape(B, J*2)]  
        if self.use_mask:
            feat.append(mask.reshape(B, J))     
        if self.use_scores:
            feat.append(scores.reshape(B, J))  

        feat = torch.cat(feat, dim=1)  

        out = self.net(feat).squeeze(1)
        return out
        