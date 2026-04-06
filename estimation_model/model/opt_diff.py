import optuna
import torch
import torch.nn as nn
import torch.optim as optim

import os, json, math, random, time
from pathlib import Path
import numpy as np
import torch

from dotenv import load_dotenv
from train_diffu_w_L_D_mpjpe import train_denoiser

load_dotenv()

# dataset
TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
TEST_SUBJECTS = ["S9", "S11"]


H36M_DIR = os.getenv("H36M_DIR")
LIFTER_CKPT = os.getenv("OPTUNA_DIFF_LIFTER_CKPT") or None
DENOISER_CKPT = os.getenv("OPTUNA_DIFF_DENOISER_CKPT") or None
KINETIC_CKPT_DIR = os.getenv("KINETIC_CKPT_DIR")
H36M_CKPT_DIR = os.getenv("H36M_CKPT_DIR")


def objective(trial):

    lambda_eps = trial.suggest_float("lambda_eps", 1.0, 10.0)
    lambda_reproj = trial.suggest_float("lambda_reproj", 1.0, 10.0)
    lambda_bone = trial.suggest_float("lambda_bone", 1.0, 10.0)
    lambda_sym = trial.suggest_float("lambda_sym", 1.0, 10.0)
    lambda_depth_var = trial.suggest_float("lambda_depth_var", 0.0, 5.0)
    c = trial.suggest_float("c", 5.0, 10.0)
    target_std = trial.suggest_float("target_z_std", 0.5, 1.0)
    drop_rate=0.1
    clamp_known = False
    T = 100
    conf_thr = 0.3
    seed = 42

    learning_rate = trial.suggest_float("learning_rate", 1e-7, 2e-5, log=True)



    batch_size = 4096
    grad_clip = 1.0
    dataset_human36m=False
    
    return train_denoiser(
        train_txt_path=os.getenv("KINETIC_50PER_FULLPOSE_TRAIN_PATH"), 
        val_txt_path=os.getenv("KINETIC_10PER_FULLPOSE_VAL_PATH"), 
        lifter_checkpoint=LIFTER_CKPT, 
        output_dir=KINETIC_CKPT_DIR, 
        denoiser_checkpoint=DENOISER_CKPT, 
        lambda_eps=lambda_eps,
        lambda_reproj=lambda_reproj,
        lambda_bone=lambda_bone,
        lambda_sym=lambda_sym,
        lambda_depth_var=lambda_depth_var,
        c=c,
        target_std=target_std,
        drop_rate=drop_rate,
        clamp_known=clamp_known,
        T=T,
        conf_thr=conf_thr,
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_clip=grad_clip,
        dataset_human36m=dataset_human36m,
        is_optuna=True,
        num_epochs=20,
        optuna_trial_num=trial.number,
    )


if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Search...")

    study = optuna.create_study(direction="minimize")
    
    study.optimize(objective, n_trials=30)

    print("OPTUNA STUDY COMPLETE!")
    print(f"Best P-MPJPE: {study.best_value:.2f}mm")
    print("Best Configuration:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")