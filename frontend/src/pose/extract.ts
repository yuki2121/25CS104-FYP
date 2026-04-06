export type Coco18KP = { xy: [number, number]; c: number };

import * as THREE from "three";



export function skeletonToCoco18(
    skeleton: THREE.Skeleton, 
    camera: THREE.Camera
): { xyz: [number, number, number][], c: number[] } {
    const get = (name: string) => skeleton.getBoneByName(name);

    const bones = {
        neck: get("mixamorigNeck"),
        head: get("mixamorigHead"),
        rShoulder: get("mixamorigRightShoulder"),
        rElbow: get("mixamorigRightForeArm"),
        rWrist: get("mixamorigRightHand"),
        lShoulder: get("mixamorigLeftShoulder"),
        lElbow: get("mixamorigLeftForeArm"),
        lWrist: get("mixamorigLeftHand"),
        rHip: get("mixamorigRightUpLeg"),
        rKnee: get("mixamorigRightLeg"),
        rAnkle: get("mixamorigRightFoot"),
        lHip: get("mixamorigLeftUpLeg"),
        lKnee: get("mixamorigLeftLeg"),
        lAnkle: get("mixamorigLeftFoot"),
    };

    const tmp = new THREE.Vector3();
    

    const getCameraSpaceXYZ = (b?: THREE.Bone | null): { pos: [number, number, number], c: number } => {
        if (!b) return { pos: [0, 0, 0], c: 0 };
        b.getWorldPosition(tmp);
        tmp.applyMatrix4(camera.matrixWorldInverse);
        return { pos: [tmp.x, tmp.y, tmp.z], c: 1 };
    };

    const rawSkele = [
        getCameraSpaceXYZ(bones.head),      
        getCameraSpaceXYZ(bones.neck),      
        getCameraSpaceXYZ(bones.rShoulder), 
        getCameraSpaceXYZ(bones.rElbow),    
        getCameraSpaceXYZ(bones.rWrist),    
        getCameraSpaceXYZ(bones.lShoulder), 
        getCameraSpaceXYZ(bones.lElbow),    
        getCameraSpaceXYZ(bones.lWrist),    
        getCameraSpaceXYZ(bones.rHip),      
        getCameraSpaceXYZ(bones.rKnee),     
        getCameraSpaceXYZ(bones.rAnkle),    
        getCameraSpaceXYZ(bones.lHip),      
        getCameraSpaceXYZ(bones.lKnee),     
        getCameraSpaceXYZ(bones.lAnkle),    
        { pos: [0,0,0], c: 0 },             
        { pos: [0,0,0], c: 0 },             
        { pos: [0,0,0], c: 0 },             
        { pos: [0,0,0], c: 0 }              
    ];

    const result = { xyz: [] as Array<[number, number, number]>, c: [] as number[] };



    rawSkele.forEach(kp => {
        result.xyz.push([
            kp.pos[0] ,
            kp.pos[1] ,
            kp.pos[2] 
        ]);
        result.c.push(kp.c);
    });

    return result;
}