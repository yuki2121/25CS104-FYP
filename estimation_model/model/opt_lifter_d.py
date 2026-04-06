import optuna
import os
from dotenv import load_dotenv
from train_lifter_with_d_ckpt_mpjpe import train_lifter



load_dotenv()


TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
TEST_SUBJECTS = ["S9", "S11"]

H36M_DIR = os.getenv("H36M_DIR")
LIFTER_CKPT = os.getenv("OPTUNA_LIFTER_LIFTER_CKPT")or None
DENOISER_CKPT = os.getenv("OPTUNA_LIFTER_DENOISER_CKPT")or None
KINETIC_CKPT_DIR = os.getenv("KINETIC_CKPT_DIR")
H36M_CKPT_DIR = os.getenv("H36M_CKPT_DIR")
KINETIC_TRAIN_PATH = os.getenv("KINETIC_10PER_FULLPOSE_TRAIN_PATH")
KINETIC_VAL_PATH = os.getenv("KINETIC_10PER_FULLPOSE_VAL_PATH")



def objective(trial):

    lambda_2d = trial.suggest_float("lambda_2d", 1.0, 50.0)
    lambda_3d = trial.suggest_float("lambda_3d", 0.5, 20.0)
    lambda_bone = trial.suggest_float("lambda_bone", 1.0, 10.0)
    lambda_sym = trial.suggest_float("lambda_sym", 1.0, 10.0)
    lambda_adv = trial.suggest_float("lambda_adv", 0.0, 10)
    lambda_depth_var = trial.suggest_float("lambda_depth_var", 0.0, 10)
    c = trial.suggest_float("c", 2, 15)
    target_std = trial.suggest_float("target_z_std", 0.2, 1)
    

    lr = trial.suggest_float("lr", 1e-6, 2e-4, log=True)


    dataset_h36m = False

    batch_size = 4096
    grad_clip = 1.0
    d_steps = 1
    assume_normalized_inputs=False

        
    return train_lifter(
        train_txt_path=KINETIC_TRAIN_PATH, 
        val_txt_path=KINETIC_VAL_PATH,
        lifter_checkpoint=LIFTER_CKPT, 
        discriminator_checkpoint=DENOISER_CKPT, 
        output_dir = KINETIC_CKPT_DIR,
        num_epochs=20, 
        learning_rate_g=lr, 
        learning_rate_d=lr, 
        batch_size=batch_size, 
        log_interval=200, 
        grad_clip=grad_clip, 
        d_steps=d_steps, 
        dataset_human36m=dataset_h36m,
        lambda_2d = lambda_2d,
        lambda_3d = lambda_3d,
        lambda_bone = lambda_bone,
        lambda_sym =  lambda_sym,
        lambda_adv = lambda_adv,
        lambda_depth_var = lambda_depth_var,
        c= c,
        target_std = target_std,
        random_seed = 42,
        is_optuna = True,
        optuna_trial = trial.number
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