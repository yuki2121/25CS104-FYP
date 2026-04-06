import torch


def mpjpe_torch(pred, gt):
    return torch.norm(pred - gt, dim=2).mean()

def n_mpjpe_torch(pred, gt, eps=1e-8):
    pred_centered = pred - pred[:, 0:1, :]
    gt_centered = gt - gt[:, 0:1, :]
    
    pred_scale = torch.norm(pred_centered, dim=2).mean(dim=1, keepdim=True)
    gt_scale = torch.norm(gt_centered, dim=2).mean(dim=1, keepdim=True)
    
    scale_factor = gt_scale / (pred_scale + eps)
    pred_scaled = pred_centered * scale_factor.unsqueeze(2)
    
    return torch.norm(pred_scaled - gt_centered, dim=2).mean()

def p_mpjpe_torch(pred, gt):
    p = pred.clone()
    g = gt.clone()
    
    p = p - p.mean(dim=1, keepdim=True)
    g = g - g.mean(dim=1, keepdim=True)

    norm_p = torch.norm(p, dim=(1, 2), keepdim=True)
    norm_g = torch.norm(g, dim=(1, 2), keepdim=True)

    p = torch.where(norm_p > 0, p / norm_p, p)
    g = torch.where(norm_g > 0, g / norm_g, g)

    K = torch.bmm(g.transpose(1, 2), p)

    U, S, V = torch.svd(K)
    R = torch.bmm(U, V.transpose(1, 2))
    p_rotated = torch.bmm(p, R.transpose(1, 2))

    p_rotated = p_rotated * norm_g
    g = g * norm_g

    errors = torch.norm(p_rotated - g, dim=2).mean(dim=1)
    
    return errors.mean()





