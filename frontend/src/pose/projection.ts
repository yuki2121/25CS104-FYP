// obsoleted
import * as THREE from "three";

export function worldToCanvasPx(world: THREE.Vector3, camera: THREE.Camera, canvas: HTMLCanvasElement) {
    const ndc = world.clone().project(camera); 
    const rect = canvas.getBoundingClientRect();

    const x = (ndc.x * 0.5 + 0.5) * rect.width;
    const y = (-ndc.y * 0.5 + 0.5) * rect.height;

    const visible = ndc.z >= -1 && ndc.z <= 1 && x >= 0 && x <= rect.width && y >= 0 && y <= rect.height;

    return { x, y, z: ndc.z, visible };
}
