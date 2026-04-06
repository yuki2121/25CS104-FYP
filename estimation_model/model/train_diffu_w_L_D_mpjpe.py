import os, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import load_kinetic_dataset, load_h36m_dataset, load_h36m_coco_dataset
from lifter import LifterMLP
from geometry import perspective_projection
from losses import masked_huber_2d, symmetry_loss, bone_len_consistency_loss, eps_loss,depth_variance_loss
from occlusion import random_drop_mask, clamp_known_joints
from normalization import normalize_coco18_torch, root_center_coco18

from diffusion import Denoiser, GaussianDiffusion
import warnings

from evaluation import mpjpe_torch, n_mpjpe_torch, p_mpjpe_torch
from shared.keypoints_order import COCO18_EDGES, H36M17_EDGES, SYMMETRY_PAIRS, H36M17_SYMMETRY_PAIRS
from seed import set_seed, seed_worker
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
TEST_SUBJECTS = ["S9", "S11"]

H36M_DIR = os.getenv("H36M_DIR")
LIFTER_CKPT = os.getenv("TRAIN_DIFF_LIFTER_CKPT")or None
DENOISER_CKPT = os.getenv("TRAIN_DIFF_DENOISER_CKPT")or None
KINETIC_CKPT_DIR = os.getenv("KINETIC_CKPT_DIR")
H36M_CKPT_DIR = os.getenv("H36M_CKPT_DIR")



def train_denoiser(
train_txt_path, 
val_txt_path,
lifter_checkpoint,
output_dir,
denoiser_checkpoint=None,
num_epochs=200, 
learning_rate=5e-4, 
batch_size=2048, 

T = 100,
conf_thr=0.3,
seed = 42,

grad_clip=1.0,

# weight for loss
lambda_eps=2,
lambda_reproj=5,
lambda_bone=10,
lambda_sym= 5,
lambda_depth_var=5,
# random drop joint
drop_rate=0.3, 
clamp_known = True,
dataset_human36m=False,
c = 5,
is_optuna=False,
optuna_trial_num=None,
target_std = 0.2
):
    
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not dataset_human36m:
        print("Training on kinetics with pelvis normalization.")
        train_dataset = load_kinetic_dataset(train_txt_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=seed_worker,generator=g)

        # val_dataset = load_kinetic_dataset(val_txt_path)
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker,generator=g)
        
        val_dataset = load_h36m_coco_dataset(H36M_DIR, TEST_SUBJECTS)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker,generator=g)
    else:
        print("Training on Human36M with root-centered normalization.")
        train_dataset = load_h36m_dataset(H36M_DIR, TRAIN_SUBJECTS)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=seed_worker,generator=g)

        val_dataset = load_h36m_dataset(H36M_DIR, TEST_SUBJECTS)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker,generator=g)

    if dataset_human36m:
        joint_num = 17
        edge = H36M17_EDGES
        symmetry_pairs = H36M17_SYMMETRY_PAIRS
    else:
        joint_num = 18
        edge = COCO18_EDGES
        symmetry_pairs = SYMMETRY_PAIRS



    # load lifter as teacher
    lifter = LifterMLP(joint_num=joint_num).to(device)
    lifter.load_state_dict(torch.load(lifter_checkpoint, map_location=device)["lifter"])
    lifter.eval()
    
    for p in lifter.parameters(): # freeze lifter parameters
        p.requires_grad = False

    
    # denoiser + diffusion

    if denoiser_checkpoint is not None:
        print(f"Loading denoiser checkpoint from {denoiser_checkpoint}")
        ckpt = torch.load(denoiser_checkpoint, map_location=device)
        denoiser = Denoiser(joint_num=joint_num, time_embedding_dim= 128, depth=6, dropout=0.1).to(device)
        denoiser.load_state_dict(ckpt["denoiser"])
        print(f"Resuming training from epoch {ckpt['epoch']} with previous val loss {ckpt.get('val_loss', 'N/A'):.6f}")
    else:
    
        denoiser = Denoiser(joint_num=joint_num, time_embedding_dim= 128, depth=6, dropout=0.1).to(device)
      
    diffusion = GaussianDiffusion(timesteps=T, beta_start=1e-4, beta_end=0.02, device=device).to(device)

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    best_val_loss = float('inf')
    best_val_mpjpe = float('inf')
    epoch_without_improve = 0
    stopping_patience = 5

    print(f"Using device: {device}")

    scaler = torch.amp.GradScaler('cuda')
    

    def run_epoch(loader, is_train=True):
        nonlocal optimizer
        denoiser.train(is_train)

        total_loss = 0.0
        val_mpjpe, val_n_mpjpe, val_p_mpjpe = 0.0, 0.0, 0.0
        epoch_p_mpjpe = 0.0
        steps=0

        mode = "Train" if is_train else "Val"
        pbar = tqdm(loader, desc=f"{mode} Epoch")

        for batch in pbar:
            x2d = batch['norm_keypoints'].to(device)
            scores = batch['scores'].to(device)
            mask = batch['mask'].to(device)
            root_type = batch['root_type'].to(device)
            
            if dataset_human36m:
                x2d = x2d - x2d[:, 0:1, :]  
            else:
                x2d, root_type, _, _ = normalize_coco18_torch(x2d, scores, mask, conf_thr=conf_thr)          

            if is_train and drop_rate > 0: # train with random drop joint
                drop_mask = random_drop_mask(mask, drop_rate)
            else:
                drop_mask = mask


            with torch.no_grad():
                x3d = lifter(x2d, scores, mask, root_type)  # [B,18,3]
                
                # root-centered for diffusion teacher
                if dataset_human36m:
                    x3d = x3d - x3d[:, :1, :]  
                else:
                    x3d = root_center_coco18(x3d)

            B = x3d.shape[0]
            t = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)

            with torch.amp.autocast('cuda'):            

                # add noise
                x_t, noise = diffusion.q_sample(x3d, t)
                if dataset_human36m:
                    x_t = x_t - x_t[:, :1, :]
                else:
                    x_t = root_center_coco18(x_t)
                drop_x2d = x2d * drop_mask.unsqueeze(-1)  # zero out dropped joints in 2D input
                drop_scores = scores * drop_mask  # zero out dropped joints in scores
                noise_pred = denoiser(x_t, drop_x2d, drop_scores, drop_mask, t, root_type)

                # noise loss
                loss_eps = eps_loss(noise_pred=noise_pred, noise_true=noise, scores=drop_scores, conf_thr=conf_thr, power=1.0, mask=drop_mask, missing_weight=1.0, visible_weight=0.2)

                # recover x0
                x3d_hat = diffusion.predict_x0(x_t, t, noise_pred)
                if dataset_human36m:
                    x3d_hat = x3d_hat - x3d_hat[:, :1, :]
                else:
                    x3d_hat = root_center_coco18(x3d_hat)

                # clamp known joints 
                if clamp_known:
                    x3d_hat = clamp_known_joints(x3d_hat, x3d, drop_mask)

                # projection loss
                x2d_hat = perspective_projection(x3d_hat, c=c)
                if dataset_human36m:
                    x2d_hat = x2d_hat - x2d_hat[:, :1, :]
                else:
                    x2d_hat, root_type,_,_ = normalize_coco18_torch(x2d_hat, scores, drop_mask, conf_thr=conf_thr)

                loss_reproj = masked_huber_2d(x2d_hat, x2d, drop_mask)

                # otherlosses
                loss_bone = bone_len_consistency_loss(x3d_hat, x3d, drop_mask, edge)
                loss_sym = symmetry_loss(x3d_hat, drop_mask, symmetry_pairs)
                loss_depth_var = depth_variance_loss(x3d_hat, drop_mask, target_std=target_std)


                loss = lambda_eps * loss_eps + lambda_reproj * loss_reproj + lambda_bone * loss_bone + lambda_sym * loss_sym  + lambda_depth_var * loss_depth_var

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)
                
                # Step the optimizer and update the scaler
                scaler.step(optimizer)
                scaler.update()

            if not is_train and steps == 0:
                with torch.no_grad():
                    gt_3d = batch['pose3d'].to(device)
                    if dataset_human36m:
                        gt_3d = gt_3d - gt_3d[:, :1, :]
                    else:
                        gt_3d = root_center_coco18(gt_3d)
                    
                    lifter_guess = lifter(x2d, scores, mask, root_type)
                    if dataset_human36m:
                        lifter_guess = lifter_guess - lifter_guess[:, :1, :]
                    else:
                        lifter_guess = root_center_coco18(lifter_guess)
                    
                    refine_t = torch.full((gt_3d.shape[0],), 50, device=device, dtype=torch.long)
                    x_sample, _ = diffusion.q_sample(lifter_guess, refine_t)


                    for step in reversed(range(50)): 
                        t_s = torch.full((gt_3d.shape[0],), step, device=device, dtype=torch.long)
                        n_p = denoiser(x_sample, x2d*drop_mask.unsqueeze(-1), scores*drop_mask, drop_mask, t_s, root_type)
                        x_sample = diffusion.p_sample(x_sample, t_s, n_p)
                        

                    if dataset_human36m:
                        x_sample = x_sample - x_sample[:, :1, :]
                    else:
                        x_sample = root_center_coco18(x_sample)
                        
                    if dataset_human36m:
                        epoch_p_mpjpe = p_mpjpe_torch(x_sample, gt_3d).item()
                    else:
                        COMMON_COCO_BODY = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
                        epoch_p_mpjpe = p_mpjpe_torch(x_sample[:, COMMON_COCO_BODY], gt_3d[:, COMMON_COCO_BODY]).item()

            pbar.set_postfix({"loss": loss.item(), "p_mpjpe": epoch_p_mpjpe if not is_train else 0.0})
            total_loss += loss.item()
            steps += 1

        return total_loss / steps, 0.0, 0.0, epoch_p_mpjpe


    for epoch in range(num_epochs):
        train_loss,_,_,_ = run_epoch(train_dataloader, is_train=True)
        val_loss, val_mpjpe, val_n_mpjpe, val_p_mpjpe = run_epoch(val_dataloader, is_train=False)

        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MPJPE: {val_mpjpe:.2f}, Val N-MPJPE: {val_n_mpjpe:.2f}, Val P-MPJPE: {val_p_mpjpe:.2f}")

        # checkpointing
        if val_p_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_p_mpjpe
            epoch_without_improve = 0
            ckpt = {
                "denoiser": denoiser.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_p_mpjpe": val_p_mpjpe,
                "seed": seed,
                "T": T,
                "weights": dict(lambda_eps=lambda_eps, lambda_reproj=lambda_reproj, lambda_bone=lambda_bone, lambda_sym=lambda_sym, lambda_depth_var=lambda_depth_var)
            }

            if dataset_human36m:
                output_subdir = Path(output_dir) / "human36m"
                if is_optuna and optuna_trial_num is not None:
                    best_path = Path(output_subdir) / f"denoiser_h36m_opt_trial_{optuna_trial_num}_best.pth"
                else:
                    best_path = Path(output_subdir) / f"denoiser_h36m_best_epoch_{epoch+1}.pth"
            else:
                if is_optuna and optuna_trial_num is not None:
                    best_path = Path(output_dir) / f"denoiser_opt_trial_{optuna_trial_num}_best.pth"
                else:
                    best_path = Path(output_dir) / f"denoiser_best_epoch_{epoch+1}.pth"
            torch.save(ckpt, best_path)
            print(f"New best model saved at P-MPJPE = {best_val_mpjpe:.2f}.")
        else:
            epoch_without_improve += 1
            print(f"No improvement for {epoch_without_improve} epochs.")

            if epoch_without_improve >= stopping_patience:
                print("Early stopping triggered.")
                break
    return best_val_mpjpe


if __name__ == "__main__":
    train_txt_path = os.getenv("KINETIC_10PER_FULLPOSE_TRAIN_PATH")
    val_txt_path = os.getenv("KINETIC_10PER_FULLPOSE_VAL_PATH")
    lifter_checkpoint=LIFTER_CKPT
    output_dir = KINETIC_CKPT_DIR
    denoiser_checkpoint = DENOISER_CKPT
        
    train_denoiser(
        train_txt_path=train_txt_path, 
        val_txt_path=val_txt_path, 
        lifter_checkpoint=lifter_checkpoint, 
        denoiser_checkpoint=denoiser_checkpoint,
        output_dir=output_dir, 
        dataset_human36m=True)

       