import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from normalization import select_17_joints,root_center_2d,root_center_3d,normalize_scale_with_factor, h36m_to_coco18
from dotenv import load_dotenv
import cdflib

load_dotenv()

class Pose2DDataset(Dataset):
    def __init__(self, json_paths, conf_thr= 0.3, min_valid_joints=6, prefilter=True):
        self.json_paths = [str(p) for p in json_paths]
        self.conf_thr = conf_thr
        self.min_valid_joints = min_valid_joints

        if prefilter:
            self.sample = self.json_paths
            print(f"Use prefiltered sample with {len(self.sample)} json files.")
            return

        # filter sample
        self.sample=[]
        invalid_count = 0
        for json_path in self.json_paths:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                nk = np.asarray(data['norm_keypoints'], dtype=np.float32) #[18,2]
                s = np.asarray(data['scores'], dtype=np.float32) #[18]
                # check dim
                if nk.shape != (18,2) or s.shape != (18,):
                    continue
                
                # make mask
                mask = self._make_mask(nk, s)
                if mask.sum() >= self.min_valid_joints:
                    self.sample.append(json_path)
            except Exception as e:
                invalid_count += 1
                continue

        if len(self.sample) == 0:
            raise ValueError("No valid samples found in the provided JSON paths.")

        print(f"Loaded {len(self.sample)} valid samples. Skipped {invalid_count} invalid samples.")

    def _make_mask(self, keypoints, scores):
        # visible joints
        visible = np.logical_not(np.isclose(keypoints[:,0],0.0) & np.isclose(keypoints[:,1],0.0))
        # conf above threshold
        conf_mask = scores >= self.conf_thr

        mask = np.logical_and(visible, conf_mask)
        return mask.astype(np.float32)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        json_path = self.sample[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        nk = np.asarray(data['norm_keypoints'], dtype=np.float32) #[18,2]
        s = np.asarray(data['scores'], dtype=np.float32) #[18]
        mask = self._make_mask(nk, s) #[18]

        norm_type = data.get('norm_type', 'default')
        root_type = 0.0 if norm_type == 'pelvis' else 1.0

        nk = torch.from_numpy(nk)  # [18,2]
        s = torch.from_numpy(s)    # [18]
        mask = torch.from_numpy(mask)  # [18]
        root_type = torch.tensor(root_type, dtype=torch.float32)

        sample = {
            'norm_keypoints': nk,  # [18,2]
            'scores': s,            # [18]
            'mask': mask,           # [18]
            'root_type': root_type, # 0: pelvis, 1: mid shoulder
            'json_path': json_path
        }

        return sample

        

def getAllPaths(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(paths)} paths from {txt_path}")

    return paths

def load_kinetic_dataset(txt_path, conf_thr=0.3, min_valid_joints=6):
    json_paths = getAllPaths(txt_path)
    dataset = Pose2DDataset(json_paths, conf_thr=conf_thr, min_valid_joints=min_valid_joints)
    print(f"Dataset loaded from {txt_path} with {len(dataset)} samples.")
    return dataset

def audit_dataset(dataset):
    pelvis_count = 0
    mid_shoulder_count = 0
    joint_counts = []

    for i in range(5000):
        sample = dataset[i]
        joint_counts.append(int(sample['mask'].sum().item()))
        if sample['root_type'].item() == 0.0:
            pelvis_count += 1
        else:
            mid_shoulder_count += 1

    print(f"Pelvis normalization count: {pelvis_count}")
    print(f"Mid-shoulder normalization count: {mid_shoulder_count}")
    print("Avg valid joints:", sum(joint_counts) / len(joint_counts))
    print("Min valid joints:", min(joint_counts))
    print("Max valid joints:", max(joint_counts))


# human 36 m cdf loading

def load_h36m_d2_cdf(cdf_path):
    cdf = cdflib.CDF(cdf_path)


    pose = cdf.varget("Pose")

    pose = np.array(pose)

    if pose.ndim == 3 and pose.shape[0] == 1:
        pose = pose[0]  

    if pose.ndim != 2:
        raise ValueError(f"Unexpected Pose shape after squeeze: {pose.shape}")

    frames = pose.shape[0]
    dims = pose.shape[1]

    if dims % 2 != 0:
        raise ValueError(f"Expected even last dim for 2D pose, got {dims}")

    joints = dims // 2
    pose = pose.reshape(frames, joints, 2)

    return pose

def load_h36m_d3_cdf(cdf_path):
    cdf = cdflib.CDF(cdf_path)
    pose = np.array(cdf.varget("Pose"))

    if pose.ndim == 3 and pose.shape[0] == 1:
        pose = pose[0]  

    if pose.ndim != 2:
        raise ValueError(f"Unexpected Pose shape after squeeze: {pose.shape}")

    frames = pose.shape[0]
    dims = pose.shape[1]

    if dims % 3 != 0:
        raise ValueError(f"Expected multiple of 3 for 3D pose, got {dims}")

    joints = dims // 3
    pose = pose.reshape(frames, joints, 3)
    return pose

def load_pair(d2_path, d3_path):
    pose2d = load_h36m_d2_cdf(d2_path)  
    pose3d = load_h36m_d3_cdf(d3_path)  

    if len(pose2d) != len(pose3d):
        n = min(len(pose2d), len(pose3d))
        pose2d = pose2d[:n]
        pose3d = pose3d[:n]

    pose2d = select_17_joints(pose2d)
    pose3d = select_17_joints(pose3d)

    pose2d = root_center_2d(pose2d)
    pose3d = root_center_3d(pose3d)

    pose2d, scale = normalize_scale_with_factor(pose2d)

    return pose2d.astype(np.float32), pose3d.astype(np.float32), scale.astype(np.float32)

def load_pair_coco(d2_path, d3_path):
    pose2d = load_h36m_d2_cdf(d2_path)  
    pose3d = load_h36m_d3_cdf(d3_path)  

    if len(pose2d) != len(pose3d):
        n = min(len(pose2d), len(pose3d))
        pose2d = pose2d[:n]
        pose3d = pose3d[:n]

    pose2d = h36m_to_coco18(pose2d)
    pose3d = h36m_to_coco18(pose3d)

    return pose2d.astype(np.float32), pose3d.astype(np.float32)

class Human36MPairDataset(Dataset):
    def __init__(self, root_dir, subjects):
        self.samples_2d = []
        self.samples_3d = []
        self.scales = []

        root_dir = Path(root_dir)

        for subj in subjects:
            d2_dir = root_dir / f"Poses_D2_Positions_{subj}" / subj / "MyPoseFeatures" / "D2_Positions"
            d3_dir = root_dir / f"Poses_D3_Positions_mono_{subj}" / subj / "MyPoseFeatures" / "D3_Positions_mono"

            if not d2_dir.exists() or not d3_dir.exists():
                print(f"Skipping missing subject dirs for {subj}")
                continue

            d2_files = sorted(d2_dir.glob("*.cdf"))

            for d2_file in d2_files:
                d3_file = d3_dir / d2_file.name
                if not d3_file.exists():
                    print(f"Missing 3D file for {d2_file.name}")
                    continue

                pose2d, pose3d, scale = load_pair(d2_file, d3_file)

                self.samples_2d.append(pose2d)
                self.scales.append(scale)
                self.samples_3d.append(pose3d)

        if not self.samples_2d:
            raise RuntimeError("No samples loaded.")

        self.samples_2d = np.concatenate(self.samples_2d, axis=0)  # (N,17,2)
        self.samples_3d = np.concatenate(self.samples_3d, axis=0)  # (N,17,3)
        self.scales = np.concatenate(self.scales, axis=0)  # (N,1)

        print("Loaded 2D:", self.samples_2d.shape)
        print("Loaded 3D:", self.samples_3d.shape)

    def __len__(self):
        return len(self.samples_2d)

    def __getitem__(self, idx):
        return {
            "norm_keypoints": self.samples_2d[idx],
            "scores": np.ones(self.samples_2d[idx].shape[0], dtype=np.float32),
            "mask": np.ones(self.samples_2d[idx].shape[0], dtype=np.float32),
            "root_type": np.float32(0.0),  
            "pose3d": self.samples_3d[idx],
            "scale": self.scales[idx],
        }

class Human36MPairCOCODataset(Dataset):
    def __init__(self, root_dir, subjects):
        self.samples_2d = []
        self.samples_3d = []


        root_dir = Path(root_dir)

        for subj in subjects:
            d2_dir = root_dir / f"Poses_D2_Positions_{subj}" / subj / "MyPoseFeatures" / "D2_Positions"
            d3_dir = root_dir / f"Poses_D3_Positions_mono_{subj}" / subj / "MyPoseFeatures" / "D3_Positions_mono"

            if not d2_dir.exists() or not d3_dir.exists():
                print(f"Skipping missing subject dirs for {subj}")
                continue

            d2_files = sorted(d2_dir.glob("*.cdf"))

            for d2_file in d2_files:
                d3_file = d3_dir / d2_file.name
                if not d3_file.exists():
                    print(f"Missing 3D file for {d2_file.name}")
                    continue

                pose2d, pose3d = load_pair_coco(d2_file, d3_file)

                self.samples_2d.append(pose2d)
                self.samples_3d.append(pose3d)
  
        if not self.samples_2d:
            raise RuntimeError("No samples loaded.")

        self.samples_2d = np.concatenate(self.samples_2d, axis=0)  # (N,17,2)
        self.samples_3d = np.concatenate(self.samples_3d, axis=0)  # (N,17,3)

        print("Loaded 2D:", self.samples_2d.shape)
        print("Loaded 3D:", self.samples_3d.shape)

    def __len__(self):
        return len(self.samples_2d)

    def __getitem__(self, idx):
        pose2d = self.samples_2d[idx].astype(np.float32)   # [18,2]
        pose3d = self.samples_3d[idx].astype(np.float32)   # [18,3]

        scores = np.ones(pose2d.shape[0], dtype=np.float32)
        mask = np.ones(pose2d.shape[0], dtype=np.float32)

        scores[[16, 17]] = 0.0
        mask[[16, 17]] = 0.0

        return {
            "norm_keypoints": pose2d,
            "scores": scores,
            "mask": mask,
            "root_type": np.float32(0.0),  
            "pose3d": pose3d,
        }



def load_h36m_dataset(root_dir, subjects):
    dataset = Human36MPairDataset(root_dir, subjects)
    print(f"H36M Dataset loaded with {len(dataset)} samples.")
    return dataset

def load_h36m_coco_dataset(root_dir, subjects):
    dataset = Human36MPairCOCODataset(root_dir, subjects)
    print(f"H36M Dataset loaded with {len(dataset)} samples.")
    return dataset


if __name__ == "__main__":
    KINETIC_10PER_FULLPOSE_TRAIN_PATH = os.getenv("KINETIC_10PER_FULLPOSE_TRAIN_PATH")
    txt_path = KINETIC_10PER_FULLPOSE_TRAIN_PATH

    dataset = load_kinetic_dataset(txt_path, conf_thr=0.3, min_valid_joints=6)
    audit_dataset(dataset)