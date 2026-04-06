import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.keypoints_order import H36M17_EDGES, COCO18_EDGES



class ResBlock(nn.Module):
    def __init__(self, dim, hidden_mult):
        super(ResBlock, self).__init__()
        hidden_dim = dim * hidden_mult
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln1 = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        return self.ln1(x + h)





class LifterMLP(nn.Module):
    def __init__(self, joint_num=18, hidden_dim=1024, dropout=0.2, depth=4, hidden_mult=2):
        super(LifterMLP, self).__init__()
        self.joint_num = joint_num
        self.input_dim = joint_num * 4  +1+(joint_num-1)*2  # 2D keypoints (2)+score(1)+mask(1) per joint(18) + root type (1) + scale (1)
        self.output_dim = joint_num * 3  # 3D keypoints
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.block = nn.ModuleList([ResBlock(hidden_dim, hidden_mult) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        if joint_num == 17:
            self.edges = H36M17_EDGES
        else:
            self.edges = COCO18_EDGES


    def forward(self, keypoints_2d, scores, mask, root_type):
        B,J,_ = keypoints_2d.shape  # [B,18,2]
        assert J == self.joint_num

        bone_feats = []
        for (a, b) in self.edges:
            bone_feats.append(keypoints_2d[:, a] - keypoints_2d[:, b])
        bone_feats = torch.cat(bone_feats, dim=1) # [B, 32]

        feat = torch.cat([
            keypoints_2d.reshape(-1, J*2),  # [B,36]
            scores.reshape(-1, J),          # [B,18]
            mask.reshape(-1, J),            # [B,18]
            root_type.reshape(-1,1),        # [B,1]
            bone_feats                       # [B,36]
        ], dim=1)  # [B,75]

        h = F.gelu(self.fc1(feat)) # [B,hidden_dim]
        h = self.dropout(h)
        for block in self.block:
            h = block(h)
        h = self.dropout(h)
        out = self.fc2(h).view(B, J, 3)  # [B,54]

        return out
