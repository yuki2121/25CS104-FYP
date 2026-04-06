# eval_h36m_test_mpjpe.py

import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from occlusion import random_drop_mask, clamp_known_joints

from dataset import load_kinetic_dataset, load_h36m_dataset, load_h36m_coco_dataset
from lifter import LifterMLP
from diffusion import Denoiser
from shared.keypoints_order import COCO18_EDGES, H36M17_EDGES, SYMMETRY_PAIRS, H36M17_SYMMETRY_PAIRS
from dotenv import load_dotenv
from seed import set_seed, seed_worker
from evaluation import mpjpe_torch, n_mpjpe_torch, p_mpjpe_torch
from normalization import normalize_2d_depend_on_format, root_center_3d_depend_on_format

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

load_dotenv()

H36M_OUTPUT_DIR = os.getenv("H36M_CKPT_DIR")

def visualize_pose_comparison(pred, gt, edges, mask=None, title="3D Pose Comparison"):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def plot_skeleton(pose, edge_list, color, label, alpha=1.0, linewidth=2, vis_mask=None):
        # Plot Joints
        for idx in range(len(pose)):
            if vis_mask is not None and vis_mask[idx] == 0:
                continue
            ax.scatter(pose[idx, 0], pose[idx, 1], pose[idx, 2], color=color, s=20, alpha=alpha)

        # Plot Edges
        for i, j in edge_list:
            if vis_mask is not None:
                if vis_mask[i] == 0 or vis_mask[j] == 0:
                    continue
            ax.plot([pose[i, 0], pose[j, 0]], 
                    [pose[i, 1], pose[j, 1]], 
                    [pose[i, 2], pose[j, 2]], 
                    color=color, alpha=alpha, lw=linewidth)

        ax.plot([], [], color=color, label=label)


    plot_skeleton(gt, edges, color='black', label='Ground Truth', alpha=0.2)
    plot_skeleton(pred, edges, color='red', label='Prediction', alpha=0.8, linewidth=3, vis_mask=mask)


    all_points = np.vstack([pred, gt])
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
    mid_x, mid_y, mid_z = all_points.mean(axis=0)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.legend()
    plt.show()

    output_path = f"{H36M_OUTPUT_DIR}/{title.replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()

@torch.no_grad()
def visualize_fixed_sample(test_dataset, lifter, denoiser, device, edges, target_idx=500, do_clamp=False, test_occlusion=False, mask= None, dataset_human36m=False):
    batch = test_dataset[target_idx]

    x2d = torch.from_numpy(batch["norm_keypoints"]).unsqueeze(0).to(device, dtype=torch.float32)
    scores = torch.from_numpy(batch["scores"]).unsqueeze(0).to(device, dtype=torch.float32)
    mask = torch.from_numpy(batch["mask"]).unsqueeze(0).to(device, dtype=torch.float32)
    root_type = torch.tensor(batch["root_type"]).unsqueeze(0).to(device, dtype=torch.float32)
    gt3d = torch.from_numpy(batch["pose3d"]).unsqueeze(0).to(device, dtype=torch.float32)

    mask_fixed = mask.clone()

    if test_occlusion:
        missing_joints = [12, 13, 14, 15, 16] 
        mask_fixed[:, missing_joints] = 0.0
        x2d = x2d * mask_fixed.unsqueeze(-1)

    pred3d = predict_3d(
        lifter=lifter, x2d=x2d, scores=scores, mask=mask_fixed,
        root_type=root_type, denoiser=denoiser,
        diffusion_steps=50, do_clamp=do_clamp, 
    )
    
    pred3d = root_center_3d_depend_on_format(pred3d, dataset_human36m=dataset_human36m)
    gt3d = root_center_3d_depend_on_format(gt3d, dataset_human36m=dataset_human36m)

    p = pred3d[0] 
    g = gt3d[0]   


    p_norm = torch.norm(p)
    g_norm = torch.norm(g)
    p_scaled = p * (g_norm / (p_norm + 1e-7))

    p_np = p_scaled.cpu().numpy()
    g_np = g.cpu().numpy()


    p_np[:, 1] *= -1 
    g_np[:, 1] *= -1

    pass_mask = mask[0].cpu().numpy() if denoiser is not None else mask_fixed[0].cpu().numpy()

    visualize_pose_comparison(p_np, g_np, edges, title=f"Sample {target_idx} {('Occluded' if test_occlusion else 'Visible')} {('Baseline') if denoiser is None else 'Denoised'} {('Clamped' if do_clamp else 'Unclamped') if denoiser is not None else ''}", mask=pass_mask if test_occlusion else None)




@torch.no_grad()
def diffusion_refine(denoiser, X_init, x2d, scores, mask, root_type, start_step=20, total_T=500, do_clamp=False):
    device = X_init.device
    B = X_init.shape[0]
    x_known = X_init.clone()

    betas = torch.linspace(1e-4, 0.02, total_T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    noise = torch.randn_like(X_init)
    x = torch.sqrt(alpha_bar[start_step-1]) * X_init + torch.sqrt(1 - alpha_bar[start_step-1]) * noise

    # denoise back to 0 
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

 
@torch.no_grad()
def predict_3d(lifter, x2d, scores, mask, root_type, denoiser=None, diffusion_steps=20, do_clamp=False, dataset_human36m=False ):

    pred = lifter(x2d, scores, mask, root_type)
    pred= root_center_3d_depend_on_format(pred, dataset_human36m=dataset_human36m)
    if denoiser is not None:
        pred = diffusion_refine(
            denoiser=denoiser,
            X_init=pred,
            x2d=x2d,
            scores=scores,
            mask=mask,
            root_type=root_type,
            start_step=diffusion_steps, 
            total_T=100,  
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
def evaluate_h36m_test(
    dataloader,
    lifter,
    denoiser=None,
    device="cuda",
    diffusion_steps=50,
    test_occlusion = False,
    do_clamp = False,
    dataset_human36m = False,
):
    total_mpjpe = 0.0
    total_nmpjpe = 0.0
    total_pmpjpe = 0.0
    total_samples = 0
    total_time_ms = 0.0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i, batch in enumerate(dataloader):
        x2d = batch["norm_keypoints"].to(device=device, dtype=torch.float32)  # [B,17,2]
        scores = batch["scores"].to(device=device, dtype=torch.float32)       # [B,17]
        mask = batch["mask"].to(device=device, dtype=torch.float32)           # [B,17]
        root_type = batch["root_type"].to(device=device, dtype=torch.float32) # [B]
        gt3d = batch["pose3d"].to(device=device, dtype=torch.float32)         # [B,17,3]
        

        x2d,root_type,_,_ = normalize_2d_depend_on_format(x2d, scores, mask, root_type, conf_thr=0.3, dataset_human36m=dataset_human36m)
        gt3d = root_center_3d_depend_on_format(gt3d, dataset_human36m=dataset_human36m)


        torch.cuda.synchronize()
        start_event.record()

        if test_occlusion:
            mask = random_drop_mask(mask, 0.1)
            x2d = x2d * mask.unsqueeze(-1)


        pred3d = predict_3d(
            lifter=lifter,
            x2d=x2d,
            scores=scores,
            mask=mask,
            root_type=root_type,
            denoiser=denoiser,
            diffusion_steps=diffusion_steps,
            do_clamp = do_clamp,
            dataset_human36m=dataset_human36m,
        )

        end_event.record()
        torch.cuda.synchronize()

        if i > 0:
            total_time_ms += start_event.elapsed_time(end_event)

        B = x2d.shape[0]



        pred3d = root_center_3d_depend_on_format(pred3d, dataset_human36m=dataset_human36m)
        gt3d = root_center_3d_depend_on_format(gt3d, dataset_human36m=dataset_human36m)

        pred_norm = torch.norm(pred3d, dim=(1,2), keepdim=True)
        gt_norm = torch.norm(gt3d, dim=(1,2), keepdim=True)

        pred3d_scaled_to_gt = pred3d * (gt_norm / (pred_norm + 1e-7))


        dist_normal = torch.norm(pred3d_scaled_to_gt - gt3d, dim=2).mean(dim=1) # [B]

        pred3d_flipped = pred3d_scaled_to_gt.clone()
        pred3d_flipped[:, :, 2] *= -1 
        dist_flipped = torch.norm(pred3d_flipped - gt3d, dim=2).mean(dim=1) # [B]

        mask_flip = (dist_flipped < dist_normal).float().view(-1, 1, 1)
        best_pred = (1 - mask_flip) * pred3d_scaled_to_gt + mask_flip * pred3d_flipped


        if dataset_human36m:
            err_mpjpe = mpjpe_torch(best_pred, gt3d)
            err_nmpjpe = n_mpjpe_torch(best_pred, gt3d)
            err_pmpjpe = p_mpjpe_torch(best_pred, gt3d)
        else:
            COMMON_COCO_BODY = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            err_mpjpe = mpjpe_torch(best_pred[:, COMMON_COCO_BODY], gt3d[:, COMMON_COCO_BODY])
            err_nmpjpe = n_mpjpe_torch(best_pred[:, COMMON_COCO_BODY], gt3d[:, COMMON_COCO_BODY])
            err_pmpjpe = p_mpjpe_torch(best_pred[:, COMMON_COCO_BODY], gt3d[:, COMMON_COCO_BODY])



        total_mpjpe += err_mpjpe.item() * B
        total_nmpjpe += err_nmpjpe.item() * B
        total_pmpjpe += err_pmpjpe.item() * B
        total_samples += B


    avg_latency_per_batch = total_time_ms / (len(dataloader) - 1)
    avg_latency_per_sample = avg_latency_per_batch / B


    result = {
        "num_samples": total_samples,
        "mpjpe": total_mpjpe / max(total_samples, 1),
        "n_mpjpe": total_nmpjpe / max(total_samples, 1),
        "p_mpjpe": total_pmpjpe / max(total_samples, 1),
        "latency_ms_per_sample": avg_latency_per_sample,
    }

    return result


def save_results_csv(rows, out_csv):
    keys = ["model", "num_samples", "mpjpe", "n_mpjpe", "p_mpjpe"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    h36m_root = os.getenv("H36M_DIR")
    lifter_ckpt = os.getenv("MPJPE_LIFTER_CKPT")or None
    denoiser_ckpt = os.getenv("MPJPE_DENOISER_CKPT")or None


    test_occlusion = False
    visualization = False
    dataset_human36m = False


    test_subjects = ["S9", "S11"]

    batch_size = 2048
    diffusion_steps = 100

    if dataset_human36m:
        test_dataset = load_h36m_dataset(h36m_root, test_subjects)
        joint_num = 17
        edge = H36M17_EDGES
        symmetry_pairs = H36M17_SYMMETRY_PAIRS
    else:
        test_dataset = load_h36m_coco_dataset(h36m_root, test_subjects)
        joint_num = 18
        edge = COCO18_EDGES
        symmetry_pairs = SYMMETRY_PAIRS

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    lifter = load_lifter(lifter_ckpt, device, joint_num=joint_num)
    denoiser = load_denoiser(denoiser_ckpt, device, joint_num=joint_num)

    test_baseline = True

    print("diffusion_steps:", diffusion_steps)

    # -----------------------------
    # Baseline
    # -----------------------------
    baseline_result = None
    if test_baseline:
        baseline_result = evaluate_h36m_test(
            dataloader=test_loader,
            lifter=lifter,
            denoiser=None,
            device=device,
            diffusion_steps=diffusion_steps,
            test_occlusion=test_occlusion,
            do_clamp=False,
            dataset_human36m=dataset_human36m,
        )
        baseline_result["model"] = "baseline_lifter"
        print("Baseline:", baseline_result)

    if visualization:
        visualize_fixed_sample(test_dataset, lifter, denoiser=None, device=device, edges=edge, target_idx=1500, do_clamp=True, test_occlusion=test_occlusion, dataset_human36m=dataset_human36m)

    # -----------------------------
    # Refined
    # -----------------------------
    refined_result = evaluate_h36m_test(
        dataloader=test_loader,
        lifter=lifter,
        denoiser=denoiser,
        device=device,
        diffusion_steps=diffusion_steps,
        test_occlusion=test_occlusion,
        do_clamp=True,
        dataset_human36m=dataset_human36m,
    )
    refined_result["model"] = "lifter_plus_denoiser_clamp"
    print("Refined clamp:", refined_result)

    if visualization:
        visualize_fixed_sample(test_dataset, lifter, denoiser=denoiser, device=device, edges=edge, target_idx=1500, do_clamp=True, test_occlusion=test_occlusion, dataset_human36m=dataset_human36m)

    refined_result = evaluate_h36m_test(
        dataloader=test_loader,
        lifter=lifter,
        denoiser=denoiser,
        device=device,
        diffusion_steps=diffusion_steps,
        test_occlusion=test_occlusion,
        do_clamp=False,
        dataset_human36m=dataset_human36m,
    )
    refined_result["model"] = "lifter_plus_denoiser_unclamp"
    print("Refined unclamp:", refined_result)
    if visualization:
        visualize_fixed_sample(test_dataset, lifter, denoiser=denoiser, device=device, edges=edge, target_idx=1500, do_clamp=False, test_occlusion=test_occlusion, dataset_human36m=dataset_human36m)



if __name__ == "__main__":
    main()