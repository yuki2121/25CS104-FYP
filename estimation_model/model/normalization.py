import torch
import numpy as np
from shared.keypoints_order import H36M_32_TO_17

# coco18 joint indices
NOSE = 0
NECK = 1
RSHO = 2
LSHO = 5
RHIP = 8
LHIP = 11

def normalize_coco18_torch(keypoints_2d, scores, mask, conf_thr=0.3, eps=1e-6):
    B,J,_ = keypoints_2d.shape  
    device = keypoints_2d.device
    dtype = keypoints_2d.dtype

    # visibility flag
    vis_neck = scores[:, NECK] >= conf_thr
    vis_lhip = scores[:, LHIP] >= conf_thr
    vis_rhip = scores[:, RHIP] >= conf_thr
    vis_lsho = scores[:, LSHO] >= conf_thr
    vis_rsho = scores[:, RSHO] >= conf_thr
    vis = (scores >= conf_thr)

    if mask is not None:
        vis = vis & (mask > 0.5)

        
    vis_neck = vis[:, NECK]
    vis_lhip = vis[:, LHIP]
    vis_rhip = vis[:, RHIP]
    vis_lsho = vis[:, LSHO]
    vis_rsho = vis[:, RSHO]

    pelvis = torch.zeros(B,2, device=device, dtype=dtype) 

    vis_both_hips = vis_lhip & vis_rhip
    vis_only_lhip = vis_lhip & (~vis_rhip)
    vis_only_rhip = vis_rhip & (~vis_lhip)
    vis_no_hips = ~(vis_lhip | vis_rhip)

    pelvis[vis_both_hips] = 0.5 * (keypoints_2d[vis_both_hips, LHIP] + keypoints_2d[vis_both_hips, RHIP])
    pelvis[vis_only_lhip] = keypoints_2d[vis_only_lhip, LHIP]
    pelvis[vis_only_rhip] = keypoints_2d[vis_only_rhip, RHIP]

    if vis_no_hips.any():
        valid_joints = vis.float()
        sum_xy = valid_joints.sum(dim=1).clamp(min=1.0)
        avg = (keypoints_2d * valid_joints[..., None]).sum(dim=1) / sum_xy[:, None]
        pelvis[vis_no_hips] = avg[vis_no_hips]

    pelvis_sc = torch.zeros(B, device=device, dtype=dtype)
    pelvis_sc[vis_both_hips] = torch.minimum(scores[vis_both_hips, LHIP], scores[vis_both_hips, RHIP])
    pelvis_sc[vis_only_lhip] = scores[vis_only_lhip, LHIP]
    pelvis_sc[vis_only_rhip] = scores[vis_only_rhip, RHIP]


    pelvis_scale = torch.ones(B, device=device, dtype=dtype)
    ok_pelvis_scale = (pelvis_sc >= conf_thr) & vis_neck
    if ok_pelvis_scale.any():
        diff = pelvis[ok_pelvis_scale] - keypoints_2d[ok_pelvis_scale, NECK]
        pelvis_scale[ok_pelvis_scale] = torch.sqrt((diff ** 2).sum(dim=1)).clamp(min=eps)


    # mid-shouder
    mid_shoulder = torch.zeros(B,2, device=device, dtype=dtype)
    vis_both_sho = vis_lsho & vis_rsho
    vis_only_lsho = vis_lsho & (~vis_rsho)
    vis_only_rsho = vis_rsho & (~vis_lsho)
    vis_no_sho = ~(vis_lsho | vis_rsho)


    mid_shoulder[vis_both_sho] = 0.5 * (keypoints_2d[vis_both_sho, LSHO] + keypoints_2d[vis_both_sho, RSHO])
    mid_shoulder[vis_only_lsho] = keypoints_2d[vis_only_lsho, LSHO]
    mid_shoulder[vis_only_rsho] = keypoints_2d[vis_only_rsho, RSHO]

    if vis_no_sho.any():
        valid_joints = vis.float()
        sum_xy = valid_joints.sum(dim=1).clamp(min=1.0)
        avg = (keypoints_2d * valid_joints[..., None]).sum(dim=1) / sum_xy[:, None]
        mid_shoulder[vis_no_sho] = avg[vis_no_sho]

    midsho_sc = torch.zeros(B, device=device, dtype=dtype)
    midsho_sc[vis_both_sho] = torch.minimum(scores[vis_both_sho, LSHO], scores[vis_both_sho, RSHO])
    midsho_sc[vis_only_lsho] = scores[vis_only_lsho, LSHO]
    midsho_sc[vis_only_rsho] = scores[vis_only_rsho, RSHO]

    midsho_scale = torch.ones(B, device=device, dtype=dtype)
    ok_midsho_scale = (midsho_sc >= conf_thr) & vis_neck
    if ok_midsho_scale.any():
        diff = keypoints_2d[ok_midsho_scale, RSHO] - keypoints_2d[ok_midsho_scale, LSHO]
        midsho_scale[ok_midsho_scale] = torch.sqrt((diff ** 2).sum(dim=1)).clamp(min=eps)

    
    use_pelvis = pelvis_sc >= midsho_sc

    root = torch.where(use_pelvis[:, None], pelvis, mid_shoulder)
    scale = torch.where(use_pelvis, pelvis_scale, midsho_scale)
    root_type = torch.where(use_pelvis, torch.zeros(B, device=device), torch.ones(B, device=device))

    scale = scale.clamp(min=eps)
    x_centered = keypoints_2d - root[:, None, :]
    x_normalized = x_centered / scale[:, None, None]

    visible = vis.float()
    x_normalized = x_normalized * visible[..., None]

    return x_normalized, root_type, scale[:, None, None], root[:, None, :]



def normalize_coco18_3d_torch(pose_3d, scores, mask, conf_thr=0.3, eps=1e-6):
    B, J, C = pose_3d.shape  
    device = pose_3d.device
    dtype = pose_3d.dtype

    vis = (scores >= conf_thr)
    if mask is not None:
        vis = vis & (mask > 0.5)

    pelvis = torch.zeros(B, 3, device=device, dtype=dtype) # [B, 3]
    vis_lhip, vis_rhip = vis[:, LHIP], vis[:, RHIP]
    
    vis_both_hips = vis_lhip & vis_rhip
    vis_only_lhip = vis_lhip & (~vis_rhip)
    vis_only_rhip = vis_rhip & (~vis_lhip)
    vis_no_hips = ~(vis_lhip | vis_rhip)

    pelvis[vis_both_hips] = 0.5 * (pose_3d[vis_both_hips, LHIP] + pose_3d[vis_both_hips, RHIP])
    pelvis[vis_only_lhip] = pose_3d[vis_only_lhip, LHIP]
    pelvis[vis_only_rhip] = pose_3d[vis_only_rhip, RHIP]

    if vis_no_hips.any():
        valid_joints = vis.float()
        sum_joints = valid_joints.sum(dim=1).clamp(min=1.0)
        avg = (pose_3d * valid_joints[..., None]).sum(dim=1) / sum_joints[:, None]
        pelvis[vis_no_hips] = avg[vis_no_hips]


    pelvis_scale = torch.ones(B, device=device, dtype=dtype)
    vis_neck = vis[:, NECK]
    
    if vis_neck.any():
        diff = pelvis - pose_3d[:, NECK] # [B, 3]
        pelvis_scale = torch.norm(diff, dim=1).clamp(min=eps)

    mid_shoulder = torch.zeros(B, 3, device=device, dtype=dtype)
    vis_lsho, vis_rsho = vis[:, LSHO], vis[:, RSHO]
    
    vis_both_sho = vis_lsho & vis_rsho
    vis_only_lsho = vis_lsho & (~vis_rsho)
    vis_only_rsho = vis_rsho & (~vis_lsho)
    vis_no_sho = ~(vis_lsho | vis_rsho)

    mid_shoulder[vis_both_sho] = 0.5 * (pose_3d[vis_both_sho, LSHO] + pose_3d[vis_both_sho, RSHO])
    mid_shoulder[vis_only_lsho] = pose_3d[vis_only_lsho, LSHO]
    mid_shoulder[vis_only_rsho] = pose_3d[vis_only_rsho, RSHO]

    use_pelvis = (vis_lhip | vis_rhip)
    
    root = torch.where(use_pelvis[:, None], pelvis, mid_shoulder)
    scale = pelvis_scale 
    
    root_type = torch.where(use_pelvis, torch.zeros(B, device=device, dtype=dtype), torch.ones(B, device=device, dtype=dtype))

    x_centered = pose_3d - root[:, None, :]
    x_normalized = x_centered / scale[:, None, None]

    x_normalized = x_normalized * vis.float()[..., None]

    return x_normalized, root_type, scale[:, None, None], root[:, None, :]


"""
human 3.6 m
"""
def root_center_2d(pose2d):
    root = pose2d[:, 0:1, :]  # pelvis
    return pose2d - root


def root_center_3d(pose3d):
    root = pose3d[:, 0:1, :]
    return pose3d - root

def normalize_scale(pose2d):
    scale = np.linalg.norm(pose2d, axis=2).mean(axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return pose2d / scale[:, :, None]


def select_17_joints(pose32):
    return pose32[:, H36M_32_TO_17, :]

def normalize_scale_3d(pose3d):
    scale = np.linalg.norm(pose3d, axis=2).mean(axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return pose3d / scale[:, :, None]

def normalize_scale_with_factor(pose2d):
    scale = np.linalg.norm(pose2d, axis=2).mean(axis=1, keepdims=True)  # [F,1]
    scale = np.maximum(scale, 1e-6)
    pose2d_norm = pose2d / scale[:, :, None]  # [F,J,2]
    return pose2d_norm.astype(np.float32), scale.astype(np.float32)



def h36m_to_coco18(h36m_pose):
    is_batched = (h36m_pose.ndim == 3)
    if not is_batched:
        h36m_pose = h36m_pose[np.newaxis, ...]

    N, _, C = h36m_pose.shape
    coco = np.zeros((N, 18, C), dtype=h36m_pose.dtype)

    # head
    coco[:, 0] = h36m_pose[:, 15]  
    coco[:, 1] = 0.5 * (h36m_pose[:, 17] + h36m_pose[:, 25])  # Neck

    # Right arm
    coco[:, 2] = h36m_pose[:, 25] 
    coco[:, 3] = h36m_pose[:, 26] 
    coco[:, 4] = h36m_pose[:, 27] 

    # Left arm
    coco[:, 5] = h36m_pose[:, 17] 
    coco[:, 6] = h36m_pose[:, 18] 
    coco[:, 7] = h36m_pose[:, 19] 

    # Right leg
    coco[:, 8]  = h36m_pose[:, 1] 
    coco[:, 9]  = h36m_pose[:, 2] 
    coco[:, 10] = h36m_pose[:, 3] 

    # Left leg
    coco[:, 11] = h36m_pose[:, 6] 
    coco[:, 12] = h36m_pose[:, 7] 
    coco[:, 13] = h36m_pose[:, 8] 
    
    # Face
    coco[:, 14] = h36m_pose[:, 16] 
    coco[:, 15] = h36m_pose[:, 16]  

    return coco if is_batched else coco[0]


def root_center_coco18(pose3d, hip_indices=(8, 11), neck_idx=1):
    r_hip = pose3d[:, hip_indices[0], :]  
    l_hip = pose3d[:, hip_indices[1], :]  
    
    root = (r_hip + l_hip) / 2.0  
    
    root_magnitude = torch.norm(root, dim=1)
    hips_missing = (root_magnitude < 1e-6).unsqueeze(1) 

    neck = pose3d[:, neck_idx, :] 
    final_root = torch.where(hips_missing, neck, root) 

    centered_pose = pose3d - final_root.unsqueeze(1)
    
    return centered_pose


def normalize_2d_depend_on_format(x2d, scores=None, mask=None, root_type=None, conf_thr=0.3, dataset_human36m=False):
    if dataset_human36m:
        x2d = x2d - x2d[:, 0:1, :]  
        return x2d, root_type, None, None
    else:
        return normalize_coco18_torch(x2d, scores, mask, conf_thr=conf_thr)
    
def root_center_3d_depend_on_format(pose3d, dataset_human36m=False):
    if dataset_human36m:
        pose3d = pose3d - pose3d[:, 0:1, :]
    else:
        return root_center_3d(pose3d)