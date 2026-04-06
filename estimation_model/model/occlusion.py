import torch

def random_drop_mask(mask, drop_rate=0.3):
    rand_matrix = torch.rand_like(mask)
    keep_condition = rand_matrix >= drop_rate
    return mask * keep_condition.float()

def clamp_known_joints(x_pred, x_teacher, mask, clamp_thr=0.3):
    m = (mask > clamp_thr).float()[..., None]  # [B,J,1]
    return m * x_teacher + (1.0 - m) * x_pred