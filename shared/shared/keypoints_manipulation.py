import math

import numpy as np
from shared.keypoints_order import COCO17_IDX, JOINT_ORDER_COCO18



def _avg_xy(a, b):
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]

def convert_2d_coco17_to_coco18(keypoints_17, conf_thr=0.1):
    keypt17 = keypoints_17["keypoints"]
    sc17 = keypoints_17["keypoint_scores"]

    out_xy = []
    out_sc = []

    for joint_name in JOINT_ORDER_COCO18:
        if joint_name == "Neck":
            lsho_idx = COCO17_IDX["LShoulder"]
            rsho_idx = COCO17_IDX["RShoulder"]
            lsho = keypt17[lsho_idx]
            rsho = keypt17[rsho_idx]
            lsho_sc = sc17[lsho_idx]
            rsho_sc = sc17[rsho_idx]

            if lsho_sc < conf_thr and rsho_sc < conf_thr:
                out_xy.append([0.0, 0.0])
                out_sc.append(0.0)
            elif lsho_sc < conf_thr:
                out_xy.append(rsho)
                out_sc.append(rsho_sc)
            elif rsho_sc < conf_thr:
                out_xy.append(lsho)
                out_sc.append(lsho_sc)
            else:
                avg_pt = _avg_xy(lsho, rsho)
                out_xy.append(avg_pt)
                avg_sc = (lsho_sc + rsho_sc) / 2.0
                out_sc.append(avg_sc)
        else:
            idx = COCO17_IDX[joint_name]
            pt = keypt17[idx]
            sc = sc17[idx]
            out_xy.append(pt)
            out_sc.append(sc)

    norm_type = "pelvis"
    norm_joint, norm_joint_sc, scale = derive_2d_pelvis_and_scale({"keypoints": out_xy, "scores": out_sc}, conf_thr=conf_thr)
    if norm_joint_sc < conf_thr:
        norm_joint, norm_joint_sc, scale = derive_2d_midShoulder_and_scale({"keypoints": out_xy, "scores": out_sc}, conf_thr=conf_thr)
        norm_type = "mid_shoulder"
    norm_xy = normalize_2d_keypoints(out_xy, out_sc, norm_joint, scale, thr=conf_thr)

    return {"joint_set": "COCO-18", "keypoints": out_xy, "scores": out_sc, "norm_joint_xy": norm_joint, "norm_joint_score": norm_joint_sc, "scale": scale, "bbox": keypoints_17.get("bbox", None), "bbox_score": keypoints_17.get("bbox_score", None), "norm_keypoints": norm_xy, "norm": [0.0, 0.0], "norm_type":norm_type}

def derive_2d_pelvis_and_scale(coco18, conf_thr=0.1):
    keypt18 = coco18["keypoints"]
    sc18 = coco18["scores"]

    lhip = keypt18[11]
    rhip = keypt18[8]
    lhip_sc = sc18[11]
    rhip_sc = sc18[8]

    pelvis = [0.0, 0.0]
    scale = 1.0
    pelvis_sc = 0.0

    if lhip_sc < conf_thr and rhip_sc < conf_thr:
        valid_joints = [(keypt18[i], sc18[i]) for i in range(len(sc18)) if sc18[i] >= conf_thr]
        if valid_joints:
            avg_x = sum([pt[0] for pt, _ in valid_joints]) / len(valid_joints)
            avg_y = sum([pt[1] for pt, _ in valid_joints]) / len(valid_joints)
            pelvis = [avg_x, avg_y]
        else:
            pelvis = [0.0, 0.0]
    elif lhip_sc < conf_thr:
        pelvis = rhip
        pelvis_sc = sc18[8]
    elif rhip_sc < conf_thr:
        pelvis = lhip
        pelvis_sc = sc18[11]
    else:
        pelvis = _avg_xy(lhip, rhip)
        pelvis_sc = min(sc18[11], sc18[8])


    if pelvis_sc>= conf_thr and sc18[1] >= conf_thr:
        dx = pelvis[0] - keypt18[1][0]
        dy = pelvis[1] - keypt18[1][1]
        scale = math.sqrt(dx*dx + dy*dy)
        if scale <= 1e-6:
            scale = 1.0
    else:
        scale = 1.0

    return pelvis, pelvis_sc, float(scale)

def derive_2d_midShoulder_and_scale(coco18, conf_thr=0.1):
    keypt18 = coco18["keypoints"]
    sc18 = coco18["scores"]

    lsho = keypt18[5]
    rsho = keypt18[2]
    lsho_sc = sc18[5]
    rsho_sc = sc18[2]

    mid_shoulder = [0.0, 0.0]
    scale = 1.0
    mid_shoulder_sc = 0.0

    if lsho_sc < conf_thr and rsho_sc < conf_thr:
        valid_joints = [(keypt18[i], sc18[i]) for i in range(len(sc18)) if sc18[i] >= conf_thr]
        if valid_joints:
            avg_x = sum([pt[0] for pt, _ in valid_joints]) / len(valid_joints)
            avg_y = sum([pt[1] for pt, _ in valid_joints]) / len(valid_joints)
            mid_shoulder = [avg_x, avg_y]
            mid_shoulder_sc = float(np.mean([sc for _, sc in valid_joints]))
        else:
            mid_shoulder = [0.0, 0.0]
    elif lsho_sc < conf_thr:
        mid_shoulder = rsho
        mid_shoulder_sc = sc18[2]
    elif rsho_sc < conf_thr:
        mid_shoulder = lsho
        mid_shoulder_sc = sc18[5]
    else:
        mid_shoulder = _avg_xy(lsho, rsho)
        mid_shoulder_sc = min(sc18[5], sc18[2])


    if mid_shoulder_sc>= conf_thr:
        dx = keypt18[2][0] - keypt18[5][0]
        dy = keypt18[2][1] - keypt18[5][1]
        scale = math.sqrt(dx*dx + dy*dy)
        if scale <= 1e-6:
            scale = 1.0
    else:
        scale = 1.0

    return mid_shoulder, mid_shoulder_sc, float(scale)

def normalize_2d_keypoints(keypoints, scores, pelvis, scale, thr=0.1):
    norm_keypoints = []
    for (x, y), s in zip(keypoints, scores):
        norm_x = 0.0
        norm_y = 0.0
        if s>=thr and scale > 1e-6:
            norm_x = (x - pelvis[0]) / scale
            norm_y = (y - pelvis[1]) / scale
        norm_keypoints.append([norm_x, norm_y])
    return norm_keypoints

def normalize_3d_keypoints(keypoints, scores, root, scale, thr=0.1):
    norm_keypoints = []
    
    for (x, y, z), s in zip(keypoints, scores):
        norm_x = 0.0
        norm_y = 0.0
        norm_z = 0.0
        
        # Only normalize if the joint is visible and scale is valid
        if s >= thr and scale > 1e-6:
            norm_x = (x - root[0]) / scale
            norm_y = (y - root[1]) / scale
            norm_z = (z - root[2]) / scale
            
        norm_keypoints.append([norm_x, norm_y, norm_z])
        
    return norm_keypoints



def normalize_coco18(keypoints, scores, conf_thr=0.1):

    norm_type = "pelvis"
    norm_joint, norm_joint_sc, scale = derive_2d_pelvis_and_scale({"keypoints": keypoints, "scores": scores}, conf_thr=conf_thr)
    if norm_joint_sc < conf_thr:
        norm_joint, norm_joint_sc, scale = derive_2d_midShoulder_and_scale({"keypoints": keypoints, "scores": scores}, conf_thr=conf_thr)
        norm_type = "mid_shoulder"
    norm_xy = normalize_2d_keypoints(keypoints, scores, norm_joint, scale, thr=conf_thr)

    return norm_xy, norm_type




def _avg_xyz(pt1, pt2):
    return [(pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0, (pt1[2] + pt2[2]) / 2.0]

def derive_3d_pelvis_and_scale(coco18, conf_thr=0.1):
    keypt18 = coco18["keypoints"]
    sc18 = coco18["scores"]

    lhip = keypt18[11]
    rhip = keypt18[8]
    lhip_sc = sc18[11]
    rhip_sc = sc18[8]

    pelvis = [0.0, 0.0, 0.0]
    scale = 1.0
    pelvis_sc = 0.0

    # calculate pelvis
    if lhip_sc < conf_thr and rhip_sc < conf_thr:
        valid_joints = [(keypt18[i], sc18[i]) for i in range(len(sc18)) if sc18[i] >= conf_thr]
        if valid_joints:
            avg_x = sum([pt[0] for pt, _ in valid_joints]) / len(valid_joints)
            avg_y = sum([pt[1] for pt, _ in valid_joints]) / len(valid_joints)
            avg_z = sum([pt[2] for pt, _ in valid_joints]) / len(valid_joints)
            pelvis = [avg_x, avg_y, avg_z]
        else:
            pelvis = [0.0, 0.0, 0.0]
    elif lhip_sc < conf_thr:
        pelvis = rhip
        pelvis_sc = sc18[8]
    elif rhip_sc < conf_thr:
        pelvis = lhip
        pelvis_sc = sc18[11]
    else:
        pelvis = _avg_xyz(lhip, rhip)
        pelvis_sc = min(sc18[11], sc18[8])

    # calculate scale between pelvis and neck 
    if pelvis_sc >= conf_thr and sc18[1] >= conf_thr:
        dx = pelvis[0] - keypt18[1][0]
        dy = pelvis[1] - keypt18[1][1]
        dz = pelvis[2] - keypt18[1][2]
        scale = math.sqrt(dx*dx + dy*dy + dz*dz)
        if scale <= 1e-6:
            scale = 1.0
    else:
        scale = 1.0

    return pelvis, pelvis_sc, float(scale)

def derive_3d_midShoulder_and_scale(coco18, conf_thr=0.1):
    keypt18 = coco18["keypoints"]
    sc18 = coco18["scores"]

    lsho = keypt18[5]
    rsho = keypt18[2]
    lsho_sc = sc18[5]
    rsho_sc = sc18[2]

    mid_shoulder = [0.0, 0.0, 0.0]
    scale = 1.0
    mid_shoulder_sc = 0.0

    # 1. calcuate mid-shoulder
    if lsho_sc < conf_thr and rsho_sc < conf_thr:
        valid_joints = [(keypt18[i], sc18[i]) for i in range(len(sc18)) if sc18[i] >= conf_thr]
        if valid_joints:
            avg_x = sum([pt[0] for pt, _ in valid_joints]) / len(valid_joints)
            avg_y = sum([pt[1] for pt, _ in valid_joints]) / len(valid_joints)
            avg_z = sum([pt[2] for pt, _ in valid_joints]) / len(valid_joints)
            mid_shoulder = [avg_x, avg_y, avg_z]
            mid_shoulder_sc = float(np.mean([sc for _, sc in valid_joints]))
        else:
            mid_shoulder = [0.0, 0.0, 0.0]
    elif lsho_sc < conf_thr:
        mid_shoulder = rsho
        mid_shoulder_sc = sc18[2]
    elif rsho_sc < conf_thr:
        mid_shoulder = lsho
        mid_shoulder_sc = sc18[5]
    else:
        mid_shoulder = _avg_xyz(lsho, rsho)
        mid_shoulder_sc = min(sc18[5], sc18[2])

    # 2. calculate scale between mid-shoulder and neck
    if mid_shoulder_sc >= conf_thr:
        dx = keypt18[2][0] - keypt18[5][0]
        dy = keypt18[2][1] - keypt18[5][1]
        dz = keypt18[2][2] - keypt18[5][2]
        scale = math.sqrt(dx*dx + dy*dy + dz*dz)
        if scale <= 1e-6:
            scale = 1.0
    else:
        scale = 1.0

    return mid_shoulder, mid_shoulder_sc, float(scale)



def normalize_coco18_3d(keypoints, scores, conf_thr=0.3):

    norm_type = "pelvis"
    norm_joint, norm_joint_sc, scale = derive_3d_pelvis_and_scale({"keypoints": keypoints, "scores": scores}, conf_thr=conf_thr)
    if norm_joint_sc < conf_thr:
        norm_joint, norm_joint_sc, scale = derive_3d_midShoulder_and_scale({"keypoints": keypoints, "scores": scores}, conf_thr=conf_thr)
        norm_type = "mid_shoulder"
    norm_xy = normalize_3d_keypoints(keypoints, scores, norm_joint, scale, thr=conf_thr)

    return norm_xy, norm_type

def pose_2d_to_vector(keypoints, scores):
    vec = []
    for (x, y), s in zip(keypoints, scores):
        vec.append(x)
        vec.append(y)
        vec.append(s)
    return np.array(vec, dtype=np.float32)

def pose_3d_to_vector(keypoints, scores):
    vec = []
    for (x, y, z), s in zip(keypoints, scores):        
        vec.append(x)
        vec.append(y)
        vec.append(z)
        vec.append(s)
        
    return np.array(vec, dtype=np.float32)

def vector_to_pose_3d(vec):
    vec = np.array(vec)
    data = vec.reshape(-1, 4)
    keypoints = data[:, :3].tolist() 
    scores = data[:, 3].tolist()     
    
    return keypoints, scores