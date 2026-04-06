import os, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import load_kinetic_dataset, load_h36m_dataset, load_h36m_coco_dataset
from lifter import LifterMLP
from geometry import apply_rotation,  random_rotation_matrix, perspective_projection
from losses import masked_mse_3d,masked_huber_2d, symmetry_loss, bone_len_consistency_loss, dis_hinge_loss, gen_hinge_loss, depth_variance_loss
from seed import set_seed, seed_worker
from discriminator import Discriminator2D
from tqdm import tqdm
from shared.keypoints_order import COCO18_EDGES, H36M17_EDGES, SYMMETRY_PAIRS, H36M17_SYMMETRY_PAIRS
from evaluation import mpjpe_torch, n_mpjpe_torch, p_mpjpe_torch
from normalization import normalize_2d_depend_on_format, root_center_3d_depend_on_format
from dotenv import load_dotenv

load_dotenv()




# dataset


TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
TEST_SUBJECTS = ["S9", "S11"]


def eval_batch(lifter, norm_keypoints, scores, mask, root_type, device, R=None, c=5.0, dataset_human36m=False):
    # 1. lift
    X = lifter(norm_keypoints, scores, mask, root_type)  # [B,18,3]
    X = root_center_3d_depend_on_format(X, dataset_human36m)

    # 2. random rotation and projection
    if R is None:
        R = random_rotation_matrix(X.shape[0], device) # [B,3,3]
    Y = apply_rotation(X, R) # [B,18,3]
    Y2d_norm = perspective_projection(Y, c=c) # [B,18,2]
    Y2d_norm, root_type, _, _ = normalize_2d_depend_on_format(Y2d_norm.float(), scores.float(), mask.float(), root_type, dataset_human36m)

    # 3. lift again
    Y_tilde = lifter(Y2d_norm, scores, mask, root_type) # [B,18,3]
    Y_tilde = root_center_3d_depend_on_format(Y_tilde, dataset_human36m)
    # 4. inverse rotate back and projection 
    R_inv = R.transpose(1,2) 
    X_tilde = apply_rotation(Y_tilde, R_inv) # [B,18,3]
    X_tilde2d_norm = perspective_projection(X_tilde, c=c) # [B,18,2]
    X_tilde2d_norm, root_type, _, _ = normalize_2d_depend_on_format(X_tilde2d_norm.float(), scores.float(), mask.float(), root_type, dataset_human36m)
    
    # compute losses
    loss_val_2d = masked_huber_2d(X_tilde2d_norm, norm_keypoints, mask)
    loss_val_3d = masked_mse_3d(Y_tilde, Y, mask)


    return loss_val_2d, loss_val_3d, R, X_tilde2d_norm, Y.detach(), Y_tilde





def train_lifter(
    train_txt_path, 
    val_txt_path,
    lifter_checkpoint, 
    discriminator_checkpoint, 
    output_dir = None,
    num_epochs=200, 
    learning_rate_g=1e-06, 
    learning_rate_d=1e-06, 
    batch_size=1024, 
    log_interval=200, 
    grad_clip=1.0, 
    d_steps=1, 
    dataset_human36m=False,
    lambda_2d = 1.375,
    lambda_3d = 14.94,
    lambda_bone = 5.47,
    lambda_sym =  9.89,
    lambda_adv = 8.82,
    lambda_depth_var = 2.51,
    c= 7.0,
    target_std = 0.85,
    random_seed = 42,
    is_optuna = False,
    optuna_trial = None
):
    set_seed(random_seed) 
    g = torch.Generator()
    g.manual_seed(random_seed)
    H36M_DIR = os.getenv("H36M_DIR")
    if not dataset_human36m:
        print("Training on kinetics with pelvis normalization.")
        train_dataset = load_kinetic_dataset(train_txt_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=seed_worker,generator=g)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_human36m:
        joint_num = 17
        edge = H36M17_EDGES
        symmetry_pairs = H36M17_SYMMETRY_PAIRS
    else:
        joint_num = 18
        edge = COCO18_EDGES
        symmetry_pairs = SYMMETRY_PAIRS

    lifter = LifterMLP(joint_num=joint_num).to(device)
    if lifter_checkpoint is not None:
        state = torch.load(lifter_checkpoint, map_location=device)
        try:
            lifter.load_state_dict(state)
        except :
            lifter.load_state_dict(state["lifter"])
    lifter.train()

    discriminator = Discriminator2D(joint_num=joint_num).to(device)
    if lifter_checkpoint is not None:
        state = torch.load(lifter_checkpoint, map_location=device)
        try:
            discriminator.load_state_dict(state["discriminator"])
        except:
            discriminator.load_state_dict(torch.load(discriminator_checkpoint, map_location=device))
    else:
        discriminator.load_state_dict(torch.load(discriminator_checkpoint, map_location=device))
    discriminator.train()

    optimizer_g = torch.optim.Adam(lifter.parameters(), lr=learning_rate_g, betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.9, 0.999))

    if lifter_checkpoint is not None and "optimizer_g" in state:
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        
        # for param_group in optimizer_g.param_groups:
        #     param_group['lr'] = learning_rate_g
        # for param_group in optimizer_d.param_groups:
        #     param_group['lr'] = learning_rate_d

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[20,40,60, 80,100,120,140, 160], gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[20, 40,60, 80,100,120,140, 160], gamma=0.5)

    best_val_loss =  float('inf')
    best_p_mpjpe = float('inf')
    epoch_without_improve = 0
    stopping_patience = 10


    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        total_2d_loss = 0.0
        total_3d_loss = 0.0
        total_loss = 0.0

        epoch_step = 0

        pbar = tqdm(train_dataloader, desc=f"Train Epoch", leave=False)


        for batch in pbar:

            norm_keypoints = batch['norm_keypoints'].to(device)
            scores = batch['scores'].to(device)
            mask = batch['mask'].to(device)
            root_type = batch['root_type'].to(device)
            
            norm_keypoints, root_type, _, _ = normalize_2d_depend_on_format(norm_keypoints.float(), scores.float(), mask.float(), root_type, dataset_human36m=dataset_human36m)

            # 1. lift
            X = lifter(norm_keypoints, scores, mask, root_type)  # [B,18,3]
            X = root_center_3d_depend_on_format(X, dataset_human36m)

            X_2d_initial = perspective_projection(X, c=c)
            X_2d_initial, root_type, _, _ = normalize_2d_depend_on_format(X_2d_initial.float(), scores.float(), mask.float(), root_type, dataset_human36m=dataset_human36m)
            loss_2d_initial = masked_huber_2d(X_2d_initial, norm_keypoints, mask)


            #  50% of the time, flip Z
            if np.random.random() > 0.5:
                X_for_disc = X.clone()
                X_for_disc[:, :, 2] *= -1
            else:
                X_for_disc = X

            # 2. random rotation and projection
            R = random_rotation_matrix(X_for_disc.shape[0], device) # [B,3,3]
            Y = apply_rotation(X_for_disc, R) # [B,18,3]
            Y2d_norm = perspective_projection(Y,c=c) # [B,18,2]
            Y2d_norm, root_type, _, _ = normalize_2d_depend_on_format(Y2d_norm.float(), scores.float(), mask.float(), root_type, dataset_human36m=dataset_human36m)


            # Discriminator step
            for p in discriminator.parameters():
                p.requires_grad_(True)
            for _ in range(d_steps):
                optimizer_d.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda'):
                    logits_real = discriminator(norm_keypoints, mask, scores)  
                    logits_fake = discriminator(Y2d_norm.detach(), mask, scores)  

                    loss_d = dis_hinge_loss(logits_real, logits_fake)
                    # print("loss_d", loss_d.item())

                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()

            

            for p in discriminator.parameters():
                p.requires_grad_(False)

            optimizer_g.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                # 3. lift again
                Y_tilde = lifter(Y2d_norm, scores, mask, root_type) # [B,18,3]
                Y_tilde = root_center_3d_depend_on_format(Y_tilde, dataset_human36m)

                # 4. inverse rotate back and projection 
                R_inv = R.transpose(1,2) 
                X_tilde = apply_rotation(Y_tilde, R_inv) # [B,18,3]
                X_tilde2d_norm = perspective_projection(X_tilde, c=c) # [B,18,2]
                X_tilde2d_norm, root_type, _, _ = normalize_2d_depend_on_format(X_tilde2d_norm.float(), scores.float(), mask.float(), root_type, dataset_human36m=dataset_human36m)



                # compute losses
                loss_2d = masked_huber_2d(X_tilde2d_norm, norm_keypoints, mask)
                loss_2d = loss_2d + loss_2d_initial
                loss_3d = masked_mse_3d(Y_tilde, Y.detach(), mask)
                loss_bone = bone_len_consistency_loss(X_tilde, X.detach(), mask, edge)
                loss_sym = symmetry_loss(X, mask, symmetry_pairs)
                loss_depth_var = depth_variance_loss(X, mask, target_std=target_std)




                # adversarial loss
                logits_fake_for_g = discriminator(Y2d_norm, mask, scores)
                loss_adv = gen_hinge_loss(logits_fake_for_g)



                loss = lambda_2d * loss_2d + lambda_3d * loss_3d  + lambda_bone * loss_bone + lambda_sym * loss_sym + lambda_adv * loss_adv + lambda_depth_var * loss_depth_var 

            
            scaler_g.scale(loss).backward()
            
            # If using grad clip with scaler:
            if grad_clip is not None and grad_clip > 0.0:
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(lifter.parameters(), grad_clip)
                
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # logging
            total_2d_loss += loss_2d.item()
            total_3d_loss += loss_3d.item()
            total_loss += loss.item()


            pbar.set_postfix({
                "Avg Loss": f"{total_loss/(epoch_step+1):.6f}",
                "Avg 2D Loss": f"{total_2d_loss/(epoch_step+1):.6f}",
                "Avg 3D Loss": f"{total_3d_loss/(epoch_step+1):.6f}",
            })

            epoch_step += 1

        print(f"Epoch {epoch+1} completed. Total Avg Loss: {total_loss/epoch_step:.6f}, Total Avg 2D Loss: {total_2d_loss/epoch_step:.6f}, Total Avg 3D Loss: {total_3d_loss/epoch_step:.6f}")


        # Validation
        lifter.eval()

        val_2d_loss = 0.0
        val_3d_loss = 0.0


        val_step = 0


        val_mpjpe, val_n_mpjpe, val_p_mpjpe = 0.0, 0.0, 0.0
        val_step = 0        

        pbar_val = tqdm(val_dataloader, desc=f"Val Epoch", leave=False) 

        for batch in pbar_val:
            norm_keypoints = batch['norm_keypoints'].to(device)
            scores = batch['scores'].to(device)
            mask = batch['mask'].to(device)
            root_type = batch['root_type'].to(device)
            
            # NEW: Extract 3D Ground Truth
            pose3d = batch.get('pose3d')

            norm_keypoints,root_type,_,_ = normalize_2d_depend_on_format(norm_keypoints.float(), scores.float(), mask.float(), root_type, dataset_human36m=dataset_human36m)


            with torch.no_grad():
                
                # NEW: Calculate pure 3D prediction for MPJPE
                X = lifter(norm_keypoints, scores, mask, root_type)  # [B,18,3]
                X = root_center_3d_depend_on_format(X, dataset_human36m)
                
                if pose3d is not None:
                    pose3d = pose3d.to(device)
                    pose3d = root_center_3d_depend_on_format(pose3d, dataset_human36m)

                    if dataset_human36m:

                        val_mpjpe += mpjpe_torch(X, pose3d).item()
                        val_n_mpjpe += n_mpjpe_torch(X, pose3d).item()
                        val_p_mpjpe += p_mpjpe_torch(X, pose3d).item()
                    else:
                    
                        COMMON_COCO_BODY = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

                        val_mpjpe   += mpjpe_torch(X[:, COMMON_COCO_BODY], pose3d[:, COMMON_COCO_BODY]).item()
                        val_n_mpjpe += n_mpjpe_torch(X[:, COMMON_COCO_BODY], pose3d[:, COMMON_COCO_BODY]).item()
                        val_p_mpjpe += p_mpjpe_torch(X[:, COMMON_COCO_BODY], pose3d[:, COMMON_COCO_BODY]).item()



                loss_val_2d, loss_val_3d, R, X_tilde2d, Y, Y_tilde = eval_batch(lifter, norm_keypoints, scores, mask, root_type, device, R=None)


            val_2d_loss += loss_val_2d.item()
            val_3d_loss += loss_val_3d.item()


            val_step += 1


        avg_val_2d_loss = val_2d_loss / val_step
        avg_val_3d_loss = val_3d_loss / val_step
        

        avg_mpjpe = val_mpjpe / val_step if val_step > 0 else 0
        avg_n_mpjpe = val_n_mpjpe / val_step if val_step > 0 else 0
        avg_p_mpjpe = val_p_mpjpe / val_step if val_step > 0 else 0


        pbar_val.set_postfix({
            "Avg 2D Loss": f"{avg_val_2d_loss:.6f}",
            "Avg 3D Loss": f"{avg_val_3d_loss:.6f}",
            "P-MPJPE": f"{avg_p_mpjpe:.2f}mm" if pose3d is not None else "N/A"
        })

        print(f"Validation - Avg 2D Loss: {avg_val_2d_loss:.6f}, Avg 3D Loss: {avg_val_3d_loss:.6f}")

        if pose3d is not None:
            print(f"Metrics    - MPJPE: {avg_mpjpe:.2f}mm, N-MPJPE: {avg_n_mpjpe:.2f}mm, P-MPJPE: {avg_p_mpjpe:.2f}mm")
            
        # save epoch model
        if is_optuna and optuna_trial is not None:
            if dataset_human36m:
                epoch_model_path = Path(f"{output_dir}/human36m/lifter_discriminator_h36m_{optuna_trial}.pth")
            else:
                epoch_model_path = Path(f"{output_dir}/lifter_discriminator_{optuna_trial}.pth")
        else:
            if dataset_human36m:
                epoch_model_path = Path(f"{output_dir}/human36m/lifter_discriminator_h36m_epoch_{epoch+1}.pth")
            else:
                epoch_model_path = Path(f"{output_dir}/lifter_discriminator_epoch_{epoch+1}.pth")


        if avg_p_mpjpe < best_p_mpjpe:
            epoch_without_improve = 0
            best_p_mpjpe = avg_p_mpjpe

            torch.save({"lifter": lifter.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "epoch": epoch+1,
                        },
                            epoch_model_path)
            print(f"Saved model checkpoint to {epoch_model_path}")
        else:
            epoch_without_improve += 1
            print(f"No improvement in validation loss for {epoch_without_improve} epochs.")
            if epoch_without_improve >= stopping_patience:
                print("Early stopping triggered.")
                break

        lifter.train()
        scheduler_g.step()
        scheduler_d.step()
        
    return best_p_mpjpe
        

if __name__ == "__main__":

    train_txt_path = os.getenv("KINETIC_10PER_TRAIN_PATH")
    val_txt_path = os.getenv("KINETIC_10PER_VAL_PATH")
    lifter_checkpoint = os.getenv("TRAIN_LIFTER_LIFTER_CKPT") or None
    discriminator_checkpoint = os.getenv("TRAIN_LIFTER_DISCRIMINATOR_CKPT") or None
    
    output_dir = os.getenv("KINETIC_CKPT_DIR") or None

    
    
    train_lifter(train_txt_path, val_txt_path, lifter_checkpoint, discriminator_checkpoint, output_dir=output_dir, dataset_human36m=False)