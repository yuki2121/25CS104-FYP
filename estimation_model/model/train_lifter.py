import os, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import load_kinetic_dataset, load_h36m_dataset, load_h36m_coco_dataset
from lifter import LifterMLP
from geometry import apply_rotation, orthographic_projection, random_rotation_matrix
from losses import masked_mse_2d, masked_mse_3d,masked_huber_2d, fix_denominstor_total_err_2d, fix_denominstor_total_err_3d, bone_length_loss, z_regularizer, bone_mean_anchor_loss, symmetry_loss, bone_len_consistency_loss, dis_hinge_loss, gen_hinge_loss, depth_variance_loss
from occlusion import random_drop_mask
from discriminator import Discriminator2D
from normalization import normalize_coco18_torch
from shared.keypoints_order import COCO18_EDGES, H36M17_EDGES, SYMMETRY_PAIRS, H36M17_SYMMETRY_PAIRS
from dotenv import load_dotenv

load_dotenv()

output_dir = os.getenv("KINETIC_CKPT_DIR") or None
train_txt_path = os.getenv("KINETIC_10PER_TRAIN_PATH")
val_txt_path = os.getenv("KINETIC_10PER_VAL_PATH")
H36M_DIR = os.getenv("H36M_DIR")

TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
TEST_SUBJECTS = ["S9", "S11"]

def eval_batch(lifter, norm_keypoints, scores, mask, root_type,scale, device,R= None):
    # 1. lift
    X = lifter(norm_keypoints, scores, mask, root_type, scale)  # [B,18,3]

    # 2. random rotation and projection
    if R is None:
        R = random_rotation_matrix(X.shape[0], device) # [B,3,3]
    Y = apply_rotation(X, R) # [B,18,3]
    Y2d = orthographic_projection(Y) # [B,18,2]
    # Y2d_norm, _, _, _ = normalize_coco18_torch(Y2d, scores, mask)

    # 3. lift again
    Y_tilde = lifter(Y2d, scores, mask, root_type, scale) # [B,18,3]
    # 4. inverse rotate back and projection 
    R_inv = R.transpose(1,2) 
    X_tilde = apply_rotation(Y_tilde, R_inv) # [B,18,3]
    X_tilde2d = orthographic_projection(X_tilde) # [B,18,2]
    # X_tilde2d_norm, _, _, _ = normalize_coco18_torch(X_tilde2d, scores, mask)
    # compute losses
    loss_val_2d = masked_huber_2d(X_tilde2d, norm_keypoints, mask)
    loss_val_3d = masked_mse_3d(Y_tilde, Y, mask)


    return loss_val_2d, loss_val_3d, R, X_tilde2d, Y.detach(), Y_tilde





def train_lifter(train_txt_path, val_txt_path, num_epochs=50, learning_rate_g=2e-4, batch_size=256, log_interval=200, lambda_2d=50.0, lambda_3d=1.0, grad_clip=1.0, d_steps=1, learning_rate_d=2e-4, dataset_human36m=False):
    if not dataset_human36m:
        print("Training on kinetics with pelvis normalization.")
        train_dataset = load_kinetic_dataset(train_txt_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = load_kinetic_dataset(val_txt_path)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        print("Training on Human36M with root-centered normalization.")
        train_dataset = load_h36m_dataset(H36M_DIR, TRAIN_SUBJECTS)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = load_h36m_dataset(H36M_DIR, TEST_SUBJECTS)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


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
    lifter.train()

    # discriminator = Discriminator2D().to(device)
    # discriminator.train()

    optimizer_g = torch.optim.Adam(lifter.parameters(), lr=learning_rate_g, betas=(0.9, 0.999))
    # optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.9, 0.999))

    
    best_val_loss =  float('inf')
    epoch_without_improve = 0
    stopping_patience = 5



    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        total_2d_loss = 0.0
        total_3d_loss = 0.0
        total_loss = 0.0

        epoch_step = 0

        for batch in train_dataloader:
            # print("Batch sanity check:")
            # print("norm_keypoints shape", batch['norm_keypoints'].shape)
            # print("scores shape", batch['scores'].shape)
            # print("mask shape", batch['mask'].shape)
            # print("mask sum per sample", batch['mask'].sum(dim=1))

            norm_keypoints = batch['norm_keypoints'].to(device)
            scores = batch['scores'].to(device)
            mask = batch['mask'].to(device)
            root_type = batch['root_type'].to(device)
            scale = batch['scale'].to(device)

            

            # 1. lift
            X = lifter(norm_keypoints, scores, mask, root_type, scale)  # [B,18,3]
            X = X - X[:, 0:1, :]
            # print("Output shape", X.shape)
            # print("Output sample", X[0])

            # 2. random rotation and projection
            R = random_rotation_matrix(X.shape[0], device) # [B,3,3]
            Y = apply_rotation(X, R) # [B,18,3]
            Y2d = orthographic_projection(Y) # [B,18,2]
            # normalize fake 2D keypoints for discriminator
            # Y2d_norm, _, _, _ = normalize_coco18_torch(Y2d, scores, mask) 





            # 3. lift again
            Y_tilde = lifter(Y2d, scores, mask, root_type, scale) # [B,18,3]

            # 4. inverse rotate back and projection 
            R_inv = R.transpose(1,2) 
            X_tilde = apply_rotation(Y_tilde, R_inv) # [B,18,3]
            X_tilde2d = orthographic_projection(X_tilde) # [B,18,2]
            # X_tilde2d_norm, _, _, _ = normalize_coco18_torch(X_tilde2d, scores, mask)

            # compute losses
            loss_2d = masked_huber_2d(X_tilde2d, norm_keypoints, mask)
            loss_3d = masked_mse_3d(Y_tilde, Y.detach(), mask)
            loss_z = z_regularizer(X, mask)
            loss_bone = bone_len_consistency_loss(X_tilde, X.detach(), mask, edge)
            loss_sym = symmetry_loss(X, mask, symmetry_pairs)
            loss_depth_var = depth_variance_loss(X, mask)
            # loss_bone_anchor = bone_mean_anchor_loss(X, mask, COCO18_EDGES, target=1.0)
            # print(f"2D Loss: {loss_2d.item():.6f}, 3D Loss: {loss_3d.item():.6f}")

            # adversarial loss
            # logits_fake_for_g = discriminator(Y2d, mask, scores)
            # loss_adv = gen_hinge_loss(logits_fake_for_g)

            loss = lambda_2d * loss_2d + lambda_3d * loss_3d + 1 * loss_z + 5 * loss_bone + 0.1 * loss_sym + 5 * loss_depth_var 


            
            optimizer_g.zero_grad()
            loss.backward()

            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(lifter.parameters(), grad_clip)

            optimizer_g.step()

            # logging
            total_2d_loss += loss_2d.item()
            total_3d_loss += loss_3d.item()
            total_loss += loss.item()

            # if epoch_step % log_interval == 0:
            #     print("x2d abs max:", norm_keypoints.abs().max().item())
            #     avg_2d_loss = total_2d_loss / (epoch_step + 1)
            #     avg_3d_loss = total_3d_loss / (epoch_step + 1)
            #     avg_loss = total_loss / (epoch_step + 1)
            #     print(f"Step {epoch_step}, Avg Loss: {avg_loss:.6f}, Avg 2D Loss: {avg_2d_loss:.6f}, Avg 3D Loss: {avg_3d_loss:.6f}")

            epoch_step += 1

        print(f"Epoch {epoch+1} completed. Total Avg Loss: {total_loss/epoch_step:.6f}, Total Avg 2D Loss: {total_2d_loss/epoch_step:.6f}, Total Avg 3D Loss: {total_3d_loss/epoch_step:.6f}")


        # Validation
        lifter.eval()

        val_2d_loss = 0.0
        val_3d_loss = 0.0
        val_2d_loss_10 = 0.0
        val_3d_loss_10 = 0.0
        val_2d_loss_30 = 0.0
        val_3d_loss_30 = 0.0
        val_2d_loss_50 = 0.0
        val_3d_loss_50 = 0.0

        fixed_2d = 0.0
        fixed_3d = 0.0
        fixed_2d_10 = 0.0
        fixed_3d_10 = 0.0
        fixed_2d_30 = 0.0
        fixed_3d_30 = 0.0
        fixed_2d_50 = 0.0
        fixed_3d_50 = 0.0

        val_step = 0

         

        torch.manual_seed(42)

        for batch in val_dataloader:
            norm_keypoints = batch['norm_keypoints'].to(device)
            scores = batch['scores'].to(device)
            mask = batch['mask'].to(device)
            root_type = batch['root_type'].to(device)
            scale = batch['scale'].to(device)
            with torch.no_grad():
                
                loss_val_2d, loss_val_3d, R, X_tilde2d, Y, Y_tilde = eval_batch(lifter, norm_keypoints, scores, mask, root_type, scale, device)

                # with mask drop
                mask_10 = random_drop_mask(mask, drop_rate=0.1)
                mask_30 = random_drop_mask(mask, drop_rate=0.3)
                mask_50 = random_drop_mask(mask, drop_rate=0.5)

                # print("mask mean:", mask.mean().item(), "mask_50 mean:", mask_50.mean().item())


                loss_val_2d_10, loss_val_3d_10, _, X_tilde2d_10, Y_10, Y_tilde_10 = eval_batch(lifter, norm_keypoints, scores, mask_10, root_type, scale, device, R)
                loss_val_2d_30, loss_val_3d_30, _, X_tilde2d_30, Y_30, Y_tilde_30 = eval_batch(lifter, norm_keypoints, scores, mask_30, root_type, scale, device, R)
                loss_val_2d_50, loss_val_3d_50, _, X_tilde2d_50, Y_50, Y_tilde_50 = eval_batch(lifter, norm_keypoints, scores, mask_50, root_type, scale, device, R )

            val_2d_loss += loss_val_2d.item()
            val_3d_loss += loss_val_3d.item()
            val_2d_loss_10 += loss_val_2d_10.item()                             
            val_3d_loss_10 += loss_val_3d_10.item()
            val_2d_loss_30 += loss_val_2d_30.item()
            val_3d_loss_30 += loss_val_3d_30.item()
            val_2d_loss_50 += loss_val_2d_50.item()
            val_3d_loss_50 += loss_val_3d_50.item()
            
            fixed_2d += fix_denominstor_total_err_2d(X_tilde2d, norm_keypoints, mask, mask).item()
            fixed_3d += fix_denominstor_total_err_3d(Y_tilde, Y, mask, mask).item()
            fixed_2d_10 += fix_denominstor_total_err_2d(X_tilde2d_10, norm_keypoints, mask, mask_10).item()
            fixed_3d_10 += fix_denominstor_total_err_3d(Y_tilde_10, Y_10, mask, mask_10).item()
            fixed_2d_30 += fix_denominstor_total_err_2d(X_tilde2d_30, norm_keypoints, mask, mask_30).item()
            fixed_3d_30 += fix_denominstor_total_err_3d(Y_tilde_30, Y_30, mask, mask_30).item()
            fixed_2d_50 += fix_denominstor_total_err_2d(X_tilde2d_50, norm_keypoints, mask, mask_50).item()
            fixed_3d_50 += fix_denominstor_total_err_3d(Y_tilde_50, Y_50, mask, mask_50).item()

            val_step += 1

        avg_val_2d_loss = val_2d_loss / val_step
        avg_val_3d_loss = val_3d_loss /val_step
        avg_val_2d_loss_10 = val_2d_loss_10 / val_step
        avg_val_3d_loss_10 = val_3d_loss_10 / val_step
        avg_val_2d_loss_30 = val_2d_loss_30 / val_step
        avg_val_3d_loss_30 = val_3d_loss_30 / val_step
        avg_val_2d_loss_50 = val_2d_loss_50 / val_step
        avg_val_3d_loss_50 = val_3d_loss_50 / val_step

        avg_fixed_2d = fixed_2d / val_step
        avg_fixed_3d = fixed_3d / val_step
        avg_fixed_2d_10 = fixed_2d_10 / val_step
        avg_fixed_3d_10 = fixed_3d_10 / val_step
        avg_fixed_2d_30 = fixed_2d_30 / val_step
        avg_fixed_3d_30 = fixed_3d_30 / val_step
        avg_fixed_2d_50 = fixed_2d_50 / val_step
        avg_fixed_3d_50 = fixed_3d_50 / val_step

        print(f"Validation - Avg 2D Loss: {avg_val_2d_loss:.6f}, Avg 3D Loss: {avg_val_3d_loss:.6f}, Fixed 2D: {avg_fixed_2d:.6f}, Fixed 3D: {avg_fixed_3d:.6f}")
        print(f"Validation with 10% mask drop - Avg 2D Loss: {avg_val_2d_loss_10:.6f}, Avg 3D Loss: {avg_val_3d_loss_10:.6f}, Fixed 2D: {avg_fixed_2d_10:.6f}, Fixed 3D: {avg_fixed_3d_10:.6f}")
        print(f"Validation with 30% mask drop - Avg 2D Loss: {avg_val_2d_loss_30:.6f}, Avg 3D Loss: {avg_val_3d_loss_30:.6f}, Fixed 2D: {avg_fixed_2d_30:.6f}, Fixed 3D: {avg_fixed_3d_30:.6f}")
        print(f"Validation with 50% mask drop - Avg 2D Loss: {avg_val_2d_loss_50:.6f}, Avg 3D Loss: {avg_val_3d_loss_50:.6f}, Fixed 2D: {avg_fixed_2d_50:.6f}, Fixed 3D: {avg_fixed_3d_50:.6f}")

        # save epoch model
        if dataset_human36m:
            epoch_model_path = Path(f"{output_dir}/human36m/lifter_epoch_{epoch+1}.pth")
        else:
            epoch_model_path = Path(f"{output_dir}/lifter_epoch_{epoch+1}.pth")
        if (avg_val_2d_loss + avg_val_3d_loss) < best_val_loss:
            epoch_without_improve = 0
            best_val_loss = avg_val_2d_loss + avg_val_3d_loss
            torch.save(lifter.state_dict(), epoch_model_path)
            print(f"Saved model checkpoint to {epoch_model_path}")
        else:
            epoch_without_improve += 1
            print(f"No improvement in validation loss for {epoch_without_improve} epochs.")
            if epoch_without_improve >= stopping_patience:
                print("Early stopping triggered.")
                break

        lifter.train()
        

if __name__ == "__main__":
    train_txt_path = os.getenv("KINETIC_10PER_TRAIN_PATH")
    val_txt_path = os.getenv("KINETIC_10PER_VAL_PATH")
    train_lifter(train_txt_path, val_txt_path, dataset_human36m=True)