import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader



from model.lifter import LifterMLP
from model.diffusion import Denoiser

from shared.keypoints_manipulation import  pose_3d_to_vector
from model.occlusion import random_drop_mask, clamp_known_joints
from shared.keypoints_order import COCO18_EDGES, H36M17_EDGES, SYMMETRY_PAIRS, H36M17_SYMMETRY_PAIRS
from database.db import get_all_2d_poses, insert_pose_3d_vector
from dotenv import load_dotenv

load_dotenv()




@torch.no_grad()
def diffusion_refine(denoiser, X_init, x2d, scores, mask, root_type, start_step=20, total_T=500, do_clamp=False):
    device = X_init.device
    B = X_init.shape[0]
    x_known = X_init.clone()


    betas = torch.linspace(1e-4, 0.02, total_T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)


    t_start = torch.full((B,), start_step - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(X_init)

    x = torch.sqrt(alpha_bar[start_step-1]) * X_init + torch.sqrt(1 - alpha_bar[start_step-1]) * noise


    for t in reversed(range(start_step)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = denoiser(x, x2d, scores, mask, t_tensor, root_type)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        x = coef1 * (x - coef2 * eps_pred)

        if t > 0:
            noise_t = torch.randn_like(x)
            sigma = torch.sqrt(betas[t])
            x = x + sigma * noise_t

        if do_clamp:
            x = clamp_known_joints(x, x_known, mask)

    return x

NECK_IDX = 1
RHIP_IDX = 8
LHIP_IDX = 11

def apply_dynamic_rooting(pred, root_type):

    B = pred.shape[0]
    
    mid_shoulder = pred[:, NECK_IDX:NECK_IDX+1, :]
    pelvis = 0.5 * (pred[:, RHIP_IDX:RHIP_IDX+1, :] + pred[:, LHIP_IDX:LHIP_IDX+1, :])
    
    rt = root_type.view(B, 1, 1)

    dynamic_root = (rt * mid_shoulder) + ((1.0 - rt) * pelvis)
    
    return pred - dynamic_root
 
@torch.no_grad()
def predict_3d(lifter, x2d, scores, mask, root_type, denoiser=None, diffusion_steps=20, do_clamp=False):

    pred = lifter(x2d, scores, mask, root_type)
    pred= apply_dynamic_rooting(pred, root_type)

    if denoiser is not None:
        pred = diffusion_refine(
            denoiser=denoiser,
            X_init=pred,
            x2d=x2d,
            scores=scores,
            mask=mask,
            root_type=root_type,
            start_step=diffusion_steps, 
            total_T=100  ,             
            do_clamp=do_clamp,
        )
    return pred





def load_lifter(ckpt_path, device, joint_num=17):
    lifter = LifterMLP(joint_num=joint_num).to(device)
    state = torch.load(ckpt_path, map_location=device)

    try:
        lifter.load_state_dict(state)
    except Exception:
        lifter.load_state_dict(state["lifter"])

    lifter.eval()
    return lifter


def load_denoiser(ckpt_path, device, joint_num=17):
    denoiser = Denoiser(joint_num=joint_num).to(device)
    state = torch.load(ckpt_path, map_location=device)

    if "denoiser" in state:
        denoiser.load_state_dict(state["denoiser"])
    else:
        denoiser.load_state_dict(state)

    denoiser.eval()
    return denoiser



@torch.no_grad()
def estimation_all_2d_poses(lifter_ckpt, denoiser_ckpt, diffusion_steps=100, dataset_human36m=False, do_clamp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    pose_data = get_all_2d_poses()

    if dataset_human36m:
        edges = H36M17_EDGES
        joint_num = 17
    else:
        edges = COCO18_EDGES
        joint_num = 18

    lifter = load_lifter(lifter_ckpt, device=device, joint_num=joint_num)
    denoiser = load_denoiser(denoiser_ckpt, device=device, joint_num=joint_num)

    print(pose_data[0])


    for item in pose_data:
        pose_id = item[0]
        json_data = item[1]
        keypoints_2d = torch.tensor(json_data["norm_keypoints"], dtype=torch.float32).unsqueeze(0).to(device)

        scores = torch.tensor(json_data["scores"], dtype=torch.float32).unsqueeze(0).to(device)

        mask = (scores > 0.3).float().to(device)

        root_val = 0.0 if json_data.get("norm_type", "default") == "pelvis" else 1.0
        root_type = torch.tensor([root_val], device=device, dtype=torch.float32)

        print("Shapes:", keypoints_2d.shape, scores.shape, mask.shape, root_type.shape)


        pred = predict_3d(
            lifter=lifter,
            denoiser=denoiser,
            x2d=keypoints_2d,
            scores=scores,
            mask=mask,
            root_type=root_type,
            diffusion_steps=diffusion_steps,
            do_clamp=do_clamp
        )

        pred = pred.cpu().numpy().reshape(joint_num, 3)  # [18, 3]
        vector = pose_3d_to_vector(pred, scores.squeeze(0).cpu().numpy())

        insert_pose_3d_vector(pose_id, vector)






if __name__ == "__main__":

    lifter_ckpt = os.getenv("ESTIMATION_LIFTER_CKPT")
    denoiser_ckpt = os.getenv("ESTIMATION_DENOISER_CKPT")

    estimation_all_2d_poses(
        lifter_ckpt=lifter_ckpt,
        denoiser_ckpt=denoiser_ckpt,
        diffusion_steps=100,
        dataset_human36m=False,
        do_clamp=True,
    )





    

