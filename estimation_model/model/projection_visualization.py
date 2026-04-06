import os
from dotenv import load_dotenv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


from lifter import LifterMLP
from geometry import apply_rotation, orthographic_projection, random_rotation_matrix
from dataset import load_kinetic_dataset
from normalization import normalize_coco18_torch
from diffusion import Denoiser
from occlusion import random_drop_mask, clamp_known_joints
from shared.keypoints_order import COCO18_EDGES




def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n



def rotation_from_a_to_b_np(a, b, eps=1e-8):
    a = _normalize(a, eps)
    b = _normalize(b, eps)

    v = np.cross(a, b)  # axis
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < eps:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(a, axis)
        v = _normalize(v, eps)
        # 180-degree rotation:
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + 2.0 * (K @ K)
        return R.astype(np.float32)

    # Rodrigues
    v = v / (s + eps)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + K * s + (K @ K) * (1.0 - c)
    return R.astype(np.float32)

def canonicalize_pose_3d_np(X3d, mask=None):
    X = np.asarray(X3d, dtype=np.float32).copy()
    if mask is None:
        mask = np.ones((X.shape[0],), dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    NECK = 1
    RSHO, LSHO = 2, 5
    RHIP, LHIP = 8, 11

    if mask[NECK] > 0.5:
        X -= X[NECK][None, :]
    else:
        X -= X.mean(axis=0, keepdims=True)


    have_hips = (mask[RHIP] > 0.5) and (mask[LHIP] > 0.5) and (mask[NECK] > 0.5)
    if have_hips:
        midhip = 0.5 * (X[RHIP] + X[LHIP])
        torso = (X[NECK] - midhip)  
    else:
        torso = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    target_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    R_up = rotation_from_a_to_b_np(torso, target_up)  
    X = (R_up @ X.T).T

    have_sho = (mask[RSHO] > 0.5) and (mask[LSHO] > 0.5)
    if have_sho:
        shoulder = X[RSHO] - X[LSHO]
        sx, sz = float(shoulder[0]), float(shoulder[2])
        yaw = np.arctan2(sz, sx)
        c, s = np.cos(-yaw), np.sin(-yaw)
        Ry = np.array([[ c, 0, s],
                       [ 0, 1, 0],
                       [-s, 0, c]], dtype=np.float32)
        X = (Ry @ X.T).T

        if X[RSHO, 0] < X[LSHO, 0]:
            X[:, 0] *= -1.0

    return X


def set_equal_aspect_3d(ax, X):
    xs, ys, zs = X[:, 0], X[:, 1], X[:, 2]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    cz = (z_min + z_max) / 2.0
    r = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    if r < 1e-6:
        r = 1.0

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def draw_skeleton_2d(ax, pts2d, mask=None, edges=COCO18_EDGES, title=None):
    pts2d = np.asarray(pts2d, dtype=np.float32)
    if mask is None:
        mask = np.ones((pts2d.shape[0],), dtype=np.float32)
    else:
        mask = np.asarray(mask, dtype=np.float32)

    # scatter joints (only visible)
    vis_idx = np.where(mask > 0.5)[0]
    ax.scatter(pts2d[vis_idx, 0], pts2d[vis_idx, 1], s=10)

    # draw edges
    for a, b in edges:
        if mask[a] > 0.5 and mask[b] > 0.5:
            xa, ya = pts2d[a]
            xb, yb = pts2d[b]
            ax.plot([xa, xb], [ya, yb])

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis() 
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)


def draw_skeleton_2d_overlay(ax, pts2d_a, pts2d_b, mask=None, edges=COCO18_EDGES, title=None):
    pts2d_a = np.asarray(pts2d_a, dtype=np.float32)
    pts2d_b = np.asarray(pts2d_b, dtype=np.float32)

    if mask is None:
        mask = np.ones((pts2d_a.shape[0],), dtype=np.float32)
    else:
        mask = np.asarray(mask, dtype=np.float32)


    draw_skeleton_2d(ax, pts2d_a, mask=mask, edges=edges, title=title)

    vis_idx = np.where(mask > 0.5)[0]
    ax.scatter(pts2d_b[vis_idx, 0], pts2d_b[vis_idx, 1], s=10)
    for a, b in edges:
        if mask[a] > 0.5 and mask[b] > 0.5:
            xa, ya = pts2d_b[a]
            xb, yb = pts2d_b[b]
            ax.plot([xa, xb], [ya, yb])


def draw_skeleton_3d(ax, pts3d, mask=None, edges=COCO18_EDGES, title=None):
    pts3d = np.asarray(pts3d, dtype=np.float32)
    if mask is None:
        mask = np.ones((pts3d.shape[0],), dtype=np.float32)
    else:
        mask = np.asarray(mask, dtype=np.float32)

    vis_idx = np.where(mask > 0.5)[0]
    ax.scatter(pts3d[vis_idx, 0], pts3d[vis_idx, 1], pts3d[vis_idx, 2], s=10)

    for a, b in edges:
        if mask[a] > 0.5 and mask[b] > 0.5:
            xa, ya, za = pts3d[a]
            xb, yb, zb = pts3d[b]
            ax.plot([xa, xb], [ya, yb], [za, zb])

    set_equal_aspect_3d(ax, pts3d[vis_idx] if len(vis_idx) else pts3d)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=15, azim=-60)
    ax.set_box_aspect([1.5, 1, 2])



@torch.no_grad()
def run_cycle(lifter, x2d, scores, mask, root_type, device, R=None, denoiser=None):

    X = lifter(x2d, scores, mask, root_type)

    if denoiser is not None:
        X = diffusion_refine(denoiser, X, x2d, scores, mask, root_type, start_step=100, total_T=100, do_clamp=True)
    torch.set_printoptions(profile="full")
    # print(X[0])
    if R is None:
        R = random_rotation_matrix(X.shape[0], device)
    Y = apply_rotation(X, R)                 # [B,J,3]
    Y2d = orthographic_projection(Y)         # [B,J,2]
    Y2d, root_type, _, _ = normalize_coco18_torch(Y2d, scores, mask) 
    
    Y_tilde = lifter(Y2d, scores, mask, root_type)  # [B,J,3]
    if denoiser is not None:
        Y_tilde = diffusion_refine(denoiser, Y_tilde, Y2d, scores, mask, root_type, start_step=100, total_T=100, do_clamp=True)
    X_tilde = apply_rotation(Y_tilde, R.transpose(1, 2))  # [B,J,3]
    x_rec = orthographic_projection(X_tilde)              # [B,J,2]
    x_rec, root_type,_,_ = normalize_coco18_torch(x_rec, scores, mask)
    
    return X, x_rec, Y, Y_tilde, R

def save_sample_figure(out_path, x2d, x_rec, X3d_can, X3d, mask, x2d_drop, mask_drop, title_prefix="sample", test_drop=False):

    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(1, 5, 1)
    draw_skeleton_2d(ax1, x2d, mask=mask, title=f"{title_prefix}: input 2D")

    ax2 = fig.add_subplot(1, 5, 2)
    draw_skeleton_2d(ax2, x2d_drop, mask=mask_drop, title=f"{title_prefix}: dropped 2d")

    output_mask = mask if not test_drop else np.ones_like(mask, dtype=np.float32)

    ax3 = fig.add_subplot(1, 5, 3)
    draw_skeleton_2d(ax3, x_rec, mask=output_mask, title=f"{title_prefix}: recon 2D")

    ax4 = fig.add_subplot(1, 5, 4)
    proj_from_3d = X3d[:, :2] 
    draw_skeleton_2d(ax4, proj_from_3d, mask=output_mask, title=f"{title_prefix}: proj(pred 3D)")



    ax5 = fig.add_subplot(1, 5, 5, projection="3d")
    draw_skeleton_3d(ax5, X3d_can, mask=output_mask, title=f"{title_prefix}: predicted 3D")



    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_grid_summary(out_path, samples, cols=4):
    n = len(samples)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3 * cols, 3 * rows))

    for i, s in enumerate(samples):
        ax = fig.add_subplot(rows, cols, i + 1)
        draw_skeleton_2d_overlay(ax, s["x2d"], s["xrec"], mask=s["mask"], title=f"#{i}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)






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




def main(val_txt, ckpt, out_dir, num_samples=30, batch_size=256, seed=42, denoise_ckpt=None, test_drop = True):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset/dataloader
    val_dataset = load_kinetic_dataset(val_txt)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load lifter
    lifter = LifterMLP().to(device)

    state = torch.load(ckpt, map_location=device)

    try:
        lifter.load_state_dict(state)
    except:
        lifter.load_state_dict(state["lifter"])
    lifter.eval()

    if denoise_ckpt is not None:
        denoiser = Denoiser().to(device)
        denoise_state = torch.load(denoise_ckpt, map_location=device)
        denoiser.load_state_dict(denoise_state["denoiser"])
        denoiser.eval()

    # Take one batch and pick N samples from it 
    batch = next(iter(val_loader))
    x2d = batch["norm_keypoints"].to(device)  # [B,J,2]
    scores = batch["scores"].to(device)       # [B,J]
    mask = batch["mask"].to(device)           # [B,J]
    root_type = batch["root_type"].to(device) # [B]

    test_mask = random_drop_mask(mask, drop_rate=0.3) if test_drop else mask
    x2d_test = x2d * test_mask.unsqueeze(-1) if test_drop else x2d
    scores_test = scores * test_mask if test_drop else scores

    B = x2d.shape[0]
    n = min(num_samples, B)
    idxs = list(range(B))
    random.shuffle(idxs)
    idxs = idxs[:n]

    # Run cycle for batch
    X3d, x_rec, _, _, _ = run_cycle(
        lifter,
        x2d_test,
        scores_test,
        test_mask,
        root_type,
        device,
        denoiser=denoiser if denoise_ckpt is not None else None
    )

    grid_samples = []
    for k, i in enumerate(idxs):
        x2d_i = x2d[i].detach().cpu().numpy()
        xrec_i = x_rec[i].detach().cpu().numpy()
        X3d_i = X3d[i].detach().cpu().numpy()
        mask_i = mask[i].detach().cpu().numpy()
        
        x2d_test_i = x2d_test[i].detach().cpu().numpy()
        mask_test_i = test_mask[i].detach().cpu().numpy()
        X3d_can_i = canonicalize_pose_3d_np(X3d_i, mask=mask_test_i)

        X3d_can_i = X3d_can_i.copy()
        X3d_can_i[:, [1, 2]] = X3d_can_i[:, [2, 1]]

        # Per-sample figure
        save_sample_figure(
            out_dir / f"sample_{k:02d}.png",
            x2d_i,
            xrec_i,
            X3d_can_i,
            X3d_i,
            mask_i,
            x2d_test_i,
            mask_test_i,
            title_prefix=f"sample {k:02d}",
            test_drop=test_drop,
        )

        grid_samples.append({"x2d": x2d_i, "xrec": xrec_i, "mask": mask_i})

    # Grid summary (overlay only)
    save_grid_summary(out_dir / "grid_overlay.png", grid_samples, cols=4)

    print(f"Saved {n} sample figures + grid to: {out_dir}")


if __name__ == "__main__":
    load_dotenv()
    main(
        val_txt=os.getenv("KINETIC_10PER_FULLPOSE_VAL_PATH"),
        ckpt=os.getenv("VISUALIZATION_LIFTER_CKPT")or None,
        denoise_ckpt=os.getenv("VISUALIZATION_DENOISER_CKPT")or None,
        out_dir=os.getenv("VISUALIZATION_OUT_DIR"),
        num_samples=40,
        batch_size=256,
        seed=42,
    )
