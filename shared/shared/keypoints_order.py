COCO17_IDX = {
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
}

JOINT_ORDER_COCO18 = [
    "Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist", "RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar"
]

H36M17_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (0, 15),
    (14, 16),
]

COCO18_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (0, 15),
    (14, 16), (15, 17),
]

H36M_32_TO_17 = [
    0,  
    1,  
    2,  
    3,  
    6,  
    7,  
    8,  
    12, 
    13, 
    15, 
    16, 
    17, 
    18, 
    19, 
    25, 
    26, 
    27  
]


SYMMETRY_PAIRS = [
    ((2,3), (5,6)),   
    ((3,4), (6,7)),   
    ((8,9), (11,12)), 
    ((9,10),(12,13)), 
]

H36M17_SYMMETRY_PAIRS = [
    ((1, 2), (4, 5)),     
    ((2, 3), (5, 6)),     
    ((14, 15), (11, 12)),
    ((15, 16), (12, 13)), 
]
