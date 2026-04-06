import torch
import torch.nn.functional as F


def masked_mse_2d(pred, target, mask):

    diff = pred - target  
    diff_squared = diff ** 2 
    diff_squared_sum = diff_squared.sum(dim=2) 
    masked_diff = diff_squared_sum * mask 

    mse = masked_diff.sum() / (mask.sum() + 1e-8)
    return mse

def masked_mse_3d(pred, target, mask):
    
    diff = pred - target  
    diff_squared = diff ** 2 
    diff_squared_sum = diff_squared.sum(dim=2) 
    masked_diff = diff_squared_sum * mask 

    mse = masked_diff.sum() / (mask.sum() + 1e-8)
    return mse

def masked_huber_2d(pred, target, mask, beta=1.0):
    diff = pred - target                        
    loss = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction="none")
    loss = loss.sum(dim=2)                      
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# obsoleted
def fix_denominstor_total_err_2d(X_tilde2d, norm_keypoints, mask, dropMask):
    diff = (X_tilde2d - norm_keypoints).pow(2).sum(dim=2) 
    fixed = (diff * mask).sum() / (dropMask.sum().clamp(min=1.0))
    return fixed

# obsoleted
def fix_denominstor_total_err_3d(Y_tilde, Y, mask, dropMask):
    diff = (Y_tilde - Y).pow(2).sum(dim=2) 
    fixed = (diff * mask).sum() / (dropMask.sum().clamp(min=1.0))
    return fixed

# obsoleted
def bone_length_loss(X, mask, edges):
    loss = 0.0
    count = 0
    for a, b in edges:
        valid = (mask[:, a] * mask[:, b]) > 0  
        if valid.sum() < 2:
            continue

        bone = X[valid, a] - X[valid, b]              
        length = torch.norm(bone, dim=1)              

        mean_len = length.mean()
        loss += ((length - mean_len) ** 2).mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=X.device)

    return loss / count

# obsoleted
def z_regularizer(X, mask, zmax=0.6):
    z = X[:, :, 2].abs()
    excess = (z - zmax).clamp(min=0.0)
    return (excess**2 * mask).sum() / mask.sum().clamp(min=1.0)

def depth_variance_loss(X, mask, target_std=0.2):
    z = X[:, :, 2]  # (B, J)

    #masked
    mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
    z_mean = (z * mask).sum(dim=1, keepdim=True) / mask_sum

    z_var = ((z - z_mean)**2 * mask).sum(dim=1) / mask_sum.squeeze()
    z_std = torch.sqrt(z_var + 1e-6)

    #hinge loss
    loss = torch.relu(target_std - z_std).mean()
    
    return loss

# obsoleted
def bone_per_edge_anchor_loss(X, mask, edges, target_lengths):
    total_loss = 0
    for idx, (a, b) in enumerate(edges):
        bone_vec = X[:, a] - X[:, b]
        current_len = torch.norm(bone_vec, dim=1)
        target = target_lengths[idx]
        rel_err = (current_len / target) - 1.0
        loss = F.smooth_l1_loss(rel_err, torch.zeros_like(rel_err), beta=0.1)
        total_loss += loss
    return total_loss / len(edges)


def symmetry_loss(X, mask, sym_pairs):
    loss = 0.0
    n = 0
    for (a,b), (c,d) in sym_pairs:
        v1 = (mask[:,a]*mask[:,b]) > 0
        v2 = (mask[:,c]*mask[:,d]) > 0
        v = v1 & v2
        if v.sum() < 1:
            continue
        l1 = torch.norm(X[v,a]-X[v,b], dim=1)
        l2 = torch.norm(X[v,c]-X[v,d], dim=1)
        loss += (l1 - l2).abs().mean()
        n += 1
    if n == 0:
        return torch.tensor(0.0, device=X.device)
    return loss / n

def bone_len_consistency_loss(A, B, mask, edges):
    loss = 0.0
    den = 0.0
    for a,b in edges:
        va = mask[:,a] * mask[:,b]
        la = torch.norm(A[:,a]-A[:,b], dim=1)
        lb = torch.norm(B[:,a]-B[:,b], dim=1)
        loss += (va * (la - lb).abs()).sum()
        den  += va.sum()
    return loss / den.clamp(min=1.0)

"""
discriminator loss
"""
def dis_hinge_loss(logits_real, logits_fake):
    loss_real = torch.relu(1.0 - logits_real).mean()
    loss_fake = torch.relu(1.0 + logits_fake).mean()
    return loss_real + loss_fake

def gen_hinge_loss(logits_fake):
    return -logits_fake.mean()


"""
diffusion loss
"""
def conf_weight(scores, conf_thr=0.3, power=1.0):
    # convert conf as weight
    w = (scores - conf_thr).clamp(min=0.0) / max(1e-6, (1.0 - conf_thr))
    return w.clamp(0.0, 1.0).pow(power)

def eps_loss(noise_pred, noise_true, scores, mask, conf_thr=0.3, power=1.0, missing_weight=1.0, visible_weight=0.2):
    w = conf_weight(scores, conf_thr, power)
    w = missing_weight * (1.0 - mask) + visible_weight * mask * w
    w.unsqueeze_(-1)
    return ((noise_pred - noise_true).pow(2) * w).sum() / w.sum().clamp(min=1.0)