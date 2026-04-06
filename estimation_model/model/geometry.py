import math
import torch

# rotation
def random_rotation_matrix(batch: int, device: torch.device):
    azimuth = (torch.rand(batch, device=device) * 2 * math.pi) - math.pi
    elevation = (torch.rand(batch, device=device) * (2 * math.pi / 9)) - (math.pi / 9)
    
    cos_y = torch.cos(azimuth)
    sin_y = torch.sin(azimuth)
    
    cos_x = torch.cos(elevation)
    sin_x = torch.sin(elevation)
    
    # Y-axis
    R_y = torch.zeros((batch, 3, 3), device=device)
    R_y[:, 0, 0] = cos_y
    R_y[:, 0, 2] = sin_y
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = -sin_y
    R_y[:, 2, 2] = cos_y
    
    # X-axis 
    R_x = torch.zeros((batch, 3, 3), device=device)
    R_x[:, 0, 0] = 1.0
    R_x[:, 1, 1] = cos_x
    R_x[:, 1, 2] = -sin_x
    R_x[:, 2, 1] = sin_x
    R_x[:, 2, 2] = cos_x
    
    R = torch.bmm(R_y, R_x)
    
    return R

def apply_rotation(keypoints_3d: torch.Tensor, R: torch.Tensor):
    result = torch.bmm(keypoints_3d, R.transpose(1, 2))
    return result

# projection

def orthographic_projection(keypoints_3d: torch.Tensor): #obsoleted
    keypoints_2d = keypoints_3d[:, :, :2]
    return keypoints_2d

def perspective_projection(Y, c=10.0):
    X_2d = Y[:, :, :2]
    Z = Y[:, :, 2:3]

    depth = torch.clamp(Z + c, min=1.0) 

    Y_2d_proj = (X_2d * c) / depth 
    
    return Y_2d_proj