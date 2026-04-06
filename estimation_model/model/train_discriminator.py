import os, json, math, random, time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import load_kinetic_dataset, load_h36m_dataset, load_h36m_coco_dataset
from lifter import LifterMLP
from geometry import apply_rotation, perspective_projection, random_rotation_matrix
from losses import dis_hinge_loss
from discriminator import Discriminator2D
from normalization import normalize_2d_depend_on_format, root_center_3d_depend_on_format
from shared.keypoints_order import COCO18_EDGES, H36M17_EDGES, SYMMETRY_PAIRS, H36M17_SYMMETRY_PAIRS

load_dotenv()

TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
TEST_SUBJECTS = ["S9", "S11"]


def eval_batch(lifter, discriminator, norm_keypoints, scores, mask, root_type, device,R= None, dataset_human36m=False):
    # 1. lift
    X = lifter(norm_keypoints, scores, mask, root_type)  # [B,18,3]
    X = root_center_3d_depend_on_format(X,dataset_human36m=dataset_human36m)

    # 2. random rotation and projection
    if R is None:
        R = random_rotation_matrix(X.shape[0], device) # [B,3,3]
    Y = apply_rotation(X, R) # [B,18,3]
    Y2d = perspective_projection(Y) # [B,18,2]
    Y2d_norm, root_type, _, _ = normalize_2d_depend_on_format(Y2d, scores, mask, root_type, dataset_human36m=dataset_human36m)

    logits_fake = discriminator(Y2d_norm, mask, scores)
    logits_real = discriminator(norm_keypoints, mask, scores)

    loss_d = dis_hinge_loss(logits_real, logits_fake)

    real_acc = (logits_real > 0).float().mean()
    fake_acc = (logits_fake < 0).float().mean()
    margin_real = logits_real.mean()
    margin_fake = logits_fake.mean()


    return loss_d, real_acc, fake_acc, margin_real, margin_fake





def train_discriminator(train_txt_path, val_txt_path, num_epochs=50, batch_size=256, log_interval=200, learning_rate_d=2e-4, conf_thr=0.3,c=5, lifter_checkpoint= "", d_steps=1, dataset_human36m=False, output_dir="checkpoints/discriminator"):
    if not dataset_human36m:
        print("Training on kinetics with pelvis normalization.")
        train_dataset = load_kinetic_dataset(train_txt_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataset = load_kinetic_dataset(val_txt_path)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        print("Training on Human36M with root-centered normalization.")
        train_dataset = load_h36m_dataset("/mnt/e/BaiduNetdiskDownload/human3.6m", TRAIN_SUBJECTS)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = load_h36m_dataset("/mnt/e/BaiduNetdiskDownload/human3.6m", TEST_SUBJECTS)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



    if dataset_human36m:
        joint_num = 17
        edge = H36M17_EDGES
        symmetry_pairs = H36M17_SYMMETRY_PAIRS
    else:
        joint_num = 18
        edge = COCO18_EDGES
        symmetry_pairs = SYMMETRY_PAIRS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lifter = LifterMLP(joint_num=joint_num).to(device)
    lifter.load_state_dict(torch.load(lifter_checkpoint, map_location=device))
    lifter.train()
    for p in lifter.parameters():
        p.requires_grad_(False)

    discriminator = Discriminator2D(joint_num=joint_num).to(device)
    discriminator.train()

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.9, 0.999))

    
    best_val_loss =  float('inf')
    epoch_without_improve = 0
    stopping_patience = 10



    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        discriminator.train()

        total_loss = 0.0

        epoch_step = 0

        for batch in train_dataloader:

            norm_keypoints = batch['norm_keypoints'].to(device)
            scores = batch['scores'].to(device)
            mask = batch['mask'].to(device)
            root_type = batch['root_type'].to(device)


            with torch.no_grad():

                # 1. lift
                X = lifter(norm_keypoints, scores, mask, root_type)  # [B,18,3]
                X = root_center_3d_depend_on_format(X,dataset_human36m=dataset_human36m)

                # 2. random rotation and projection
                R = random_rotation_matrix(X.shape[0], device) # [B,3,3]
                Y = apply_rotation(X, R) # [B,18,3]
                Y2d = perspective_projection(Y,c=c) # [B,18,2]
                Y2d_norm, root_type, _, _ = normalize_2d_depend_on_format(Y2d, scores, mask, root_type, dataset_human36m=dataset_human36m)



            # Discriminator step
            for _ in range(d_steps):
                logits_real = discriminator(norm_keypoints, mask, scores)  
                logits_fake = discriminator(Y2d_norm.detach(), mask, scores)  

                loss_d = dis_hinge_loss(logits_real, logits_fake)

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()


            # logging

            total_loss += loss_d.item()


            epoch_step += 1

        print(f"Epoch {epoch+1} completed. Total Avg Loss: {total_loss/epoch_step:.6f}")


        # Validation
        discriminator.eval()
        
        val_loss = 0.0
        val_real_acc = 0.0
        val_fake_acc = 0.0

        val_step = 0

         

        torch.manual_seed(42)

        for batch in val_dataloader:
            with torch.no_grad():
                norm_keypoints = batch['norm_keypoints'].to(device)
                scores = batch['scores'].to(device)
                mask = batch['mask'].to(device)
                root_type = batch['root_type'].to(device)


            
                
                loss_d, real_acc, fake_acc, margin_real, margin_fake  = eval_batch(lifter, discriminator, norm_keypoints, scores, mask, root_type, device, dataset_human36m=dataset_human36m)



            val_loss += loss_d.item()
            val_real_acc += real_acc.item()
            val_fake_acc += fake_acc.item()                             

            


            val_step += 1

        avg_val_loss_d = val_loss / val_step
        avg_val_real_acc = val_real_acc /val_step
        avg_val_fake_acc = val_fake_acc / val_step
        

        print(f"Validation - Avg Discriminator Loss: {avg_val_loss_d:.6f}, Avg Real Accuracy: {avg_val_real_acc:.6f}, Avg Fake Accuracy: {avg_val_fake_acc:.6f}")

        # save epoch model
        if dataset_human36m:
            epoch_model_path = Path(f"{output_dir}/human36m/discriminator_h36m_epoch_{epoch+1}.pth")
        else:
            epoch_model_path = Path(f"{output_dir}/discriminator_new_epoch_{epoch+1}.pth")
        if (avg_val_loss_d) < best_val_loss:
            epoch_without_improve = 0
            best_val_loss = avg_val_loss_d
            torch.save(discriminator.state_dict(), epoch_model_path)
            print(f"Saved model checkpoint to {epoch_model_path}")
        else:
            epoch_without_improve += 1
            print(f"No improvement in validation loss for {epoch_without_improve} epochs.")
            if epoch_without_improve >= stopping_patience:
                print("Early stopping triggered.")
                break

        discriminator.train()
        

if __name__ == "__main__":
    train_txt_path = os.getenv("KINETIC_10PER_TRAIN_PATH")
    val_txt_path = os.getenv("KINETIC_10PER_VAL_PATH")
    lifter_checkpoint=os.getenv("TRAIN_DISCRIMINATOR_LIFTER_CKPT")
    train_discriminator(train_txt_path, val_txt_path, lifter_checkpoint=lifter_checkpoint,output_dir=os.getenv("KINETIC_CKPT_DIR"), dataset_human36m=False)